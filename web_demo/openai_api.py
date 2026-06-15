import logging

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def vlm_openai_post(
    url: str,
    img_url: str,
    prompt: str,
    *,
    model_name: str = "",
    api_key: str = "YOUR_API_KEY",
    max_tokens: int = 512,
    temperature: float = 0.01,
) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=url)
        if not model_name:
            model_name = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            },
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        error_info = f"接口请求报错，报错信息：{str(e)}"
        logger.error(error_info)
        return error_info



def llm_openai_post(
    url: str,
    usr_message: str,
    *,
    system_prompt: str = "",
    model_name: str = "",
    api_key: str = "1",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=url)
        if not model_name:
            model_name = client.models.list().data[0].id

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_message},
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content
    except Exception as e:
        error_info = f"接口请求报错，报错信息：{str(e)}"
        logger.error(error_info)
        return error_info



if __name__ == "__main__":
    text = "将这些图片中的所有内容全部识别出来"
    image_url = "/home/cbsu/Pyproject/VLM-E-commerce/few_shot/images/air_conditioner_summary_img.jpg"

    ## 本地部署
    vlm_model_name = "gemini-2.5-pro"
    vlm_request_url = "http://159.138.85.84:8000/v1beta/openai/chat/completions"
    res = vlm_openai_post(
        vlm_request_url,
        image_url,
        text,
        api_key="AIzaSyAFgUrDom9JBXmEnqKiIJcUNI4-Mvd2E0o",
        model_name=vlm_model_name,
    )
    print(res)
