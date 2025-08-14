import torch

from . import chroma_cache


class ChromaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                # "residual_diff_threshold": (
                #     "FLOAT",
                #     {
                #         "default": 0.0,
                #         "min": 0.0,
                #         "max": 1.0,
                #         "step": 0.001,
                #         "tooltip": "Controls the tolerance for caching with lower values being more strict. Setting this to 0 disables the FBCache effect.",
                #     },
                # ),
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "step": 0.01,
                        "max": 1.0,
                        "min": 0.0,
                        "tooltip": "When to enable caching as a percentage of sampling progress. At least 0.3 is recommended because early steps are more sensitive.",
                    },
                ),
                "end": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "max": 1.0,
                    "min": 0.0,
                    "tooltip": "When to disable caching as a percentage of sampling progress.",
                }),
                "cache_interval": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "tooltip": "Number of consecutive cache hits before the cache is reset. A value of 1 means the cache is reset every other step.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Chroma Cache"

    def patch(
        self,
        model,
        start=0.0,
        end=1.0,
        cache_interval=1,
    ):
        chroma_cache.patch_get_output_data()

        model_sampling = model.get_model_object("model_sampling")
        start_sigma, end_sigma = (float(
            model_sampling.percent_to_sigma(pct)) for pct in (start, end))
        del model_sampling

        @torch.compiler.disable()
        def validate_use_cache(use_cached):
            nonlocal consecutive_cache_hits
            use_cached = use_cached and end_sigma <= current_timestep <= start_sigma
            use_cached = use_cached and consecutive_cache_hits < cache_interval
            consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
            return use_cached
        
        prev_timestep = None
        prev_input_state = None
        current_timestep = None
        consecutive_cache_hits = 0

        def reset_cache_state():
            # Resets the cache state and hits/time tracking variables.
            nonlocal prev_input_state, prev_timestep
            prev_input_state = prev_timestep = None
            chroma_cache.set_current_cache_context(
                chroma_cache.create_cache_context())

        def ensure_cache_state(model_input: torch.Tensor, timestep: float):
            # Validates the current cache state and hits/time tracking variables
            # and triggers a reset if necessary. Also updates current_timestep and
            # maintains the cache context sequence number.
            nonlocal current_timestep
            input_state = (model_input.shape, model_input.dtype, model_input.device)
            cache_context = chroma_cache.get_current_cache_context()
            # We reset when:
            need_reset = (
                # The previous timestep or input state is not set
                prev_timestep is None or
                prev_input_state is None or
                # Or dtype/device have changed
                prev_input_state[1:] != input_state[1:] or
                # Or the input state after the batch dimension has changed
                prev_input_state[0][1:] != input_state[0][1:] or
                # Or there is no cache context (in this case reset is just making a context)
                cache_context is None or
                # Or the current timestep is higher than the previous one
                timestep > prev_timestep
            )
            if need_reset:
                reset_cache_state()
            elif timestep == prev_timestep:
                # When the current timestep is the same as the previous, we assume ComfyUI has split up
                # the model evaluation into multiple chunks. In this case, we increment the sequence number.
                # Note: No need to check if cache_context is None for these branches as need_reset would be True
                #       if so.
                cache_context.sequence_num += 1
            elif timestep < prev_timestep:
                # When the timestep is less than the previous one, we can reset the context sequence number
                cache_context.sequence_num = 0
            current_timestep = timestep

        def update_cache_state(model_input: torch.Tensor, timestep: float):
            # Updates the previous timestep and input state validation variables.
            nonlocal prev_timestep, prev_input_state
            prev_timestep = timestep
            prev_input_state = (model_input.shape, model_input.dtype, model_input.device)

        model = model.clone()
        diffusion_model = model.get_model_object("diffusion_model")

        if diffusion_model.__class__.__name__ != "Chroma":
            raise ValueError(
                f"Unsupported model {diffusion_model.__class__.__name__}")

        create_patch_function = chroma_cache.create_patch_forward_orig

        patch_forward = create_patch_function(
            diffusion_model,
            validate_can_use_cache_function=validate_use_cache,
        )

        def model_unet_function_wrapper(model_function, kwargs):
            try:
                input = kwargs["input"]
                timestep = kwargs["timestep"]
                c = kwargs["c"]
                t = timestep[0].item()

                ensure_cache_state(input, t)

                with patch_forward():
                    result = model_function(input, timestep, **c)
                    update_cache_state(input, t)
                    return result
            except Exception as exc:
                reset_cache_state()
                raise exc from None

        model.set_model_unet_function_wrapper(model_unet_function_wrapper)
        return (model,)
