# comfyui-chroma-cache
ComfyUI node implementing caching optimizations for [Chroma](https://huggingface.co/lodestones/Chroma)

Inspired by techniques like TeaCache and FBCache. Unlike other models, Chroma's first block residual has low similarity between timesteps, so we refresh the cache at fixed intervals instead.

# Usage
Reasonable settings:
- start: 0.30
- end: 1.0
- caching interval: 2-4

# Acknowledgements
Cache management code is from [Comfy-WaveSpeed](https://github.com/chengzeyi/Comfy-WaveSpeed)