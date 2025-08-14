# Cache management adapted from https://github.com/chengzeyi/Comfy-WaveSpeed (MIT License)

import contextlib
import dataclasses
import unittest
from collections import defaultdict
from typing import DefaultDict, Dict

import torch


@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, list] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int))
    sequence_num: int = 0
    use_cache: bool = False

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable()
    def get_buffer(self, name):
        item = self.buffers.get(name)
        if item is None or self.sequence_num >= len(item):
            return None
        return item[self.sequence_num]

    @torch.compiler.disable()
    def set_buffer(self, name, buffer):
        curr_item = self.buffers.get(name)
        if curr_item is None:
            curr_item = []
            self.buffers[name] = curr_item
        curr_item += [None] * (self.sequence_num - len(curr_item) + 1)
        curr_item[self.sequence_num] = buffer

    def clear_buffers(self):
        self.sequence_num = 0
        self.buffers.clear()


@torch.compiler.disable()
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    return CacheContext()


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


def patch_get_output_data():
    import execution

    get_output_data = getattr(execution, "get_output_data", None)
    if get_output_data is None:
        return

    if getattr(get_output_data, "_patched", False):
        return

    def new_get_output_data(*args, **kwargs):
        out = get_output_data(*args, **kwargs)
        cache_context = get_current_cache_context()
        if cache_context is not None:
            cache_context.clear_buffers()
            set_current_cache_context(None)
        return out

    new_get_output_data._patched = True
    execution.get_output_data = new_get_output_data


@torch.compiler.disable()
def are_two_tensors_similar(t1, t2):
    return t1.shape == t2.shape


@torch.compiler.disable()
def get_can_use_cache(hidden_states, validation_function=None):
    prev_hidden_state = get_buffer("hidden_states")
    cache_context = get_current_cache_context()
    if cache_context is None or prev_hidden_state is None:
        return False
    can_use_cache = are_two_tensors_similar(hidden_states, prev_hidden_state)
    if cache_context.sequence_num > 0:
        cache_context.use_cache &= can_use_cache
    else:
        if validation_function is not None:
            can_use_cache = validation_function(can_use_cache)
        cache_context.use_cache = can_use_cache
    return cache_context.use_cache


def create_patch_forward_orig(model, *, validate_can_use_cache_function=None):
    from torch import Tensor
    from comfy.ldm.flux.model import timestep_embedding

    def call_remaining_blocks(self, blocks_replace, control, img, txt, mod_vectors, pe, attn_mask):
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                       txt=args["txt"],
                                                       vec=args["vec"],
                                                       pe=args["pe"],
                                                       attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                               "txt": txt,
                                                               "vec": double_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=double_mod,
                                     pe=pe,
                                     attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": single_mod,
                                                               "pe": pe,
                                                               "attn_mask": attn_mask},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = img.contiguous()
        return img

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                "Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        # guidance = guidance *
        distill_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = torch.cat([distill_timestep, distill_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})

        can_use_cache = get_can_use_cache(img, validation_function=validate_can_use_cache_function)

        torch._dynamo.graph_break()
        if can_use_cache:
            img = get_buffer("hidden_states")
        else:
            img = call_remaining_blocks(self, blocks_replace, control, img, txt, mod_vectors, pe, attn_mask)
            set_buffer("hidden_states", img)
        torch._dynamo.graph_break()

        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img
    
    new_forward_orig = forward_orig.__get__(model)

    @contextlib.contextmanager
    def patch_forward_orig():
        with unittest.mock.patch.object(model, "forward_orig", new_forward_orig):
            yield

    return patch_forward_orig