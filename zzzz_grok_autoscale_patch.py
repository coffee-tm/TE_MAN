from __future__ import annotations

from .Grok_Safe_PY import (
    BananaGeminiAsyncSafePyNode,
    BananaGeminiImageSafePyNode,
    BananaGrokImageSafePyNode,
    BananaGrokVideoSafePyNode,
    BananaJimengVideoSafePyNode,
    BananaSora2VideoSafePyNode,
    TEGPTImage2SafePyNode,
)


# Keep the original node ids and display names, but route execution through
# the pure-Python autoscale wrapper before entering the compiled Grok nodes.
NODE_CLASS_MAPPINGS = {
    "TE_image_pro_grok_image": BananaGrokImageSafePyNode,
    "TE_image_pro_grok_video": BananaGrokVideoSafePyNode,
    "TE_image_pro_gpt_image_2": TEGPTImage2SafePyNode,
    "TE_image_pro_banana": BananaGeminiImageSafePyNode,
    "TE_image_pro_special_async2": BananaGeminiAsyncSafePyNode,
    "TE_image_pro_jimeng_video": BananaJimengVideoSafePyNode,
    "TE_image_pro_sora2_video": BananaSora2VideoSafePyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_image_pro_grok_image": "TE MAN Grok Image",
    "TE_image_pro_grok_video": "TE MAN Grok Video",
    "TE_image_pro_gpt_image_2": "TE MAN GPT Image 2",
    "TE_image_pro_banana": "TE MAN Gemini Image",
    "TE_image_pro_special_async2": "TE MAN Gemini Async",
    "TE_image_pro_jimeng_video": "TE MAN Jimeng Video",
    "TE_image_pro_sora2_video": "TE MAN sora2 video",
}
