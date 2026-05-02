from __future__ import annotations


class TEPromptTextNode:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "build_text"
    CATEGORY = "TE MAN/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "增强提示词文本。前端支持 @ 选择下游 Grok 节点接入的图片，实际输出仍是 图1/图2/图3 文本。",
                    },
                ),
            },
        }

    def build_text(self, text: str = ""):
        return (str(text or ""),)


NODE_CLASS_MAPPINGS = {
    "TE_prompt_text": TEPromptTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_prompt_text": "TE 提示词 text",
}
