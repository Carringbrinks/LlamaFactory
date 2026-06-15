import os
import json
import base64
import gradio as gr
from copy import deepcopy
from openai import OpenAI
from css import CSS
from format_output import format_response, inverse_response


def encode_image(image_path: str) -> bytes:
    if not os.path.isfile(image_path):
        print(f"文件不存在: {image_path}")
        return ""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"读取或编码图片失败: {image_path}, 错误信息: {e}")
        return ""


def save_log(json_path, messages):
    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(messages, ensure_ascii=False) + "\n")


def clear_chat(messages_log, r1=False):
    if messages_log:
        messages_log = inverse_response(messages_log) if r1 else messages_log
        save_log(history_log_path, messages_log)
    return []


def user(message, system_message, history):
    if not history and system_message:
        history.append({"role": "system", "content": system_message})
    for x in message["files"]:
        history.append({"role": "user", "content": {"path": x}})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=True)


def bot(
    history: list,
    system_message: str,
    openai_base_url="",
    openai_api_key="",
    model_name="",
    max_tokens=2048,
    top_p=0.7,
    temperature=0.95,
    stream=False,
    r1=False,
):
    
    if system_message:
        if history[0]["role"] == "system":
            history[0]["content"] = system_message
        else:
            history.insert(0, {"role": "system", "content": system_message})
    elif history and history[0]["role"] == "system":
        history.pop(0)
    infer_history = deepcopy(history)
    # for message in infer_history:
    if r1:
        infer_history = inverse_response(infer_history)
    history.append({"role": "assistant", "content": ""})
    for message in infer_history:
        if isinstance(message["content"], tuple):
            imagse_path = message["content"][0]
            image_data = encode_image(imagse_path)
            message["content"] = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]
        
    # print(infer_history)
    llm = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    if not model_name:
        model_name = llm.models.list().data[0].id
    response_llm = llm.chat.completions.create(
        model=model_name,
        messages=infer_history,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )

    if stream:
        result = ""
        for chunk in response_llm:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
                response = format_response(result) if r1 else result
                history[-1] = {"role": "assistant", "content": response}
                yield history
    else:
        response_llm = response_llm.choices[0].message.content
        response = format_response(response_llm) if r1 else response_llm
        history[-1] = {"role": "assistant", "content": response}
        yield history


def webui():
    with gr.Blocks(css=CSS) as demo:
        with gr.Column():
            chatbot = gr.Chatbot(
                # type="messages",
                # bubble_full_width=False,
                scale=1,
                # show_copy_button=True,
                height=500,
            )
           
            with gr.Row():
                with gr.Column():
                    sys_msg = gr.Textbox(
                        "",
                        lines=3,
                        placeholder="Enter system message...",
                        label="System Prompt...",
                    )
                    with gr.Row():
                        stream_output = gr.Checkbox(
                            value=True,
                            label="Stream Output",
                            info="Whether to enable streaming output?",
                        )
                        reasoning_mode = gr.Checkbox(
                            label="Reasoning Mode",
                            info="Whether to start inference mode (the model needs to support it)",
                        )

                    # ✅ 改成 MultimodalTextbox：支持文字 + 图片上传 + 预览
                    chat_input = gr.MultimodalTextbox(
                        placeholder="Enter text or upload an image...",
                        label="User Input (Text / Image)",
                        show_label=True,
                        interactive=True,
                    )
                    clear = gr.Button(variant="primary", value="clear")

                with gr.Column():
                    url = gr.Textbox(
                        "http://localhost:8000/v1",
                        placeholder="Enter OpenAI URL",
                        label="OpenAI URL",
                    )
                    api_key = gr.Textbox(
                        "1", placeholder="Enter OpenAI Key...", label="OpenAI Key"
                    )
                    model_name = gr.Textbox(
                        "", placeholder="Enter OpenAI Model Name...", label="Model Name"
                    )
                    max_new_tokens = gr.Slider(
                        100, 8192, value=4096, step=1, label="Max_new_tokens"
                    )
                    top_p = gr.Slider(0.01, 1.0, value=0.7, step=0.01, label="Top_p")
                    temperature = gr.Slider(
                        0.01, 1.5, value=0.95, step=0.01, label="Temperature"
                    )

        # 绑定事件
        msg = (
            chat_input.submit(
                user, [chat_input, sys_msg, chatbot], [chatbot, chat_input]
            )
            .then(
                bot,
                [
                    chatbot,
                    sys_msg,
                    url,
                    api_key,
                    model_name,
                    max_new_tokens,
                    top_p,
                    temperature, 
                    stream_output,
                    reasoning_mode,
                ],
                [chatbot],
            )
            .then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        )
        clear.click(
            clear_chat,
            [chatbot, reasoning_mode],
            [chatbot],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    history_log_path = "./log_conversation_history.jsonl"
    webui().launch(share=True)
