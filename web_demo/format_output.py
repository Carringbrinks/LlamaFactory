import re

def _escape_html(text: str) -> str:
    r"""Escape HTML characters."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def format_response(text: str, lang: str="zh", escape_html: bool=True, thought_words: tuple[str, str]=("<think>", "</think>")) -> str:
    r"""Post-process the response text.

    Based on: https://huggingface.co/spaces/Lyte/DeepSeek-R1-Distill-Qwen-1.5B-Demo-GGUF/blob/main/app.py
    """
    ALERTS = {"info_thinking": {
        "en": "🌀 Thinking...",
        "ru": "🌀 Думаю...",
        "zh": "🌀 思考中...",
        "ko": "🌀 생각 중...",
        "ja": "🌀 考えています...",
    },
    "info_thought": {
        "en": "✅ Thought",
        "ru": "✅ Думать закончено",
        "zh": "✅ 思考完成",
        "ko": "✅ 생각이 완료되었습니다",
        "ja": "✅ 思考完了",
    }}
    if thought_words[0] not in text:
        return _escape_html(text) if escape_html else text

    text = text.replace(thought_words[0], "")
    result = text.split(thought_words[1], maxsplit=1)
    if len(result) == 1:
        summary = ALERTS["info_thinking"][lang]
        thought, answer = text, ""
    else:
        summary = ALERTS["info_thought"][lang]
        thought, answer = result

    if escape_html:
        thought, answer = _escape_html(thought), _escape_html(answer)

    return (
        f"<details open><summary class='thinking-summary'><span>{summary}</span></summary>\n\n"
        f"<div class='thinking-container'>\n{thought}\n</div>\n</details>{answer}"
    )


def inverse_response(history: list):
    for i, text in enumerate(history):
        if text.get("role", "") == "assistant":
            content = text.get("content", "")

            thought_match = re.search(r"<div class='thinking-container'>\n?(.*?)\n?</div>", content, re.DOTALL)
            answer_match = re.search(r"</details>\s*(.*)", content, re.DOTALL)

            if thought_match and answer_match:
                thought = thought_match.group(1).strip() 
                src_thought = f"<think>\n{thought}\n</think>"
                answer = answer_match.group(1).strip() 
            else:
                src_thought = content
                answer = ""
            history[i]["content"] = src_thought + answer
    
    return history


             

