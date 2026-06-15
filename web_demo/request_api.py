import asyncio
import base64
import logging
import os
import traceback

import httpx
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    

class LLMClient:
    def __init__(
        self,
        url: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        few_shot: bool = False,
    ):
        self.url = url
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.few_shot = few_shot
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _buile_image_part(self, img_url):
        image_part = None
        if img_url:
            if img_url.startswith("http://") or img_url.startswith("https://"):
                image_part = {"type": "image_url", "image_url": {"url": img_url}}

            elif os.path.isfile(img_url):
                # 本地图片转 base64
                image_data = encode_image(img_url)
                image_part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            else:
                raise ValueError(
                    f"图片格式不支持，仅支持base64编码和可用url， 当前传入的是{img_url}"
                )

            return image_part

    def _build_vlm_message(self, prompt: str, img_url: str = "") -> list:

        image_part = self._buile_image_part(img_url)
        content = [{"type": "text", "text": prompt}]
        if image_part:
            content.append(image_part)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ]
        return messages

    def _build_llm_message(self, usr: str, system: str = "") -> list:

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": usr},
        ]
        return messages

    def _build_payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _build_base_llm_payload(self, messages:str) -> dict:
        return {
            "model": self.model_name,
            "prompt": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _extract_response(self, data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"响应解析失败: {e}, 原始响应: {data}")
            return f"解析失败: {e}"
    
    def _extract_base_llm_response(self, data: dict) -> str:
        try:
            return data["choices"][0]["text"]
        except Exception as e:
            logger.error(f"响应解析失败: {e}, 原始响应: {data}")
            return f"解析失败: {e}"

    # ------------------- 同步请求 -------------------

    def post_llm(self, usr_message: str, system_prompt: str = "") -> str:
        messages = self._build_llm_message(usr_message, system_prompt)
        payload = self._build_payload(messages)
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload)
            # resp.raise_for_status()
            return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"post_llm 请求失败: {e}")
            return str(e)
        
    def post_llm_base(self, prompt: str = "") -> str:
        payload = self._build_base_llm_payload(prompt)
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload)
            # resp.raise_for_status()
            return self._extract_base_llm_response(resp.json())
        except Exception as e:
            logger.error(f"post_llm_base 请求失败: {e}")
            return str(e)


    def post_vlm(self, img_url: str, prompt: str) -> str:
        messages = self._build_vlm_message(prompt, img_url)
        payload = self._build_payload(messages)
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload)
            # resp.raise_for_status()
            return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"post_vlm 请求失败: {e}")
            return str(e)

    def post_vlm_as_llm(self, prompt: str) -> str:
        messages = self._build_vlm_message(prompt)
        payload = self._build_payload(messages)
        try:
            resp = requests.post(self.url, headers=self.headers, json=payload)
            # resp.raise_for_status()
            return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"post_vlm_as_llm 请求失败: {e}")
            return str(e)

    # ------------------- 异步请求 -------------------
    async def apost_llm(self, usr_message: str, system_prompt: str = "") -> str:
        messages = self._build_llm_message(usr_message, system_prompt)
        payload = self._build_payload(messages)

        try:
            async with httpx.AsyncClient(timeout=100000.0) as client:
                resp = await client.post(self.url, headers=self.headers, json=payload)
                # resp.raise_for_status()
                return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"apost_llm 请求失败: {str(e)}")
            return str(e)

    async def apost_vlm(
        self, img_url: str, prompt: str
    ) -> str:
        messages = self._build_vlm_message(prompt, img_url)
        payload = self._build_payload(messages)

        try:
            async with httpx.AsyncClient(timeout=100000.0) as client:
                resp = await client.post(self.url, headers=self.headers, json=payload)
                # resp.raise_for_status()
                return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"apost_vlm 请求失败: {str(e)}")
            return str(e)

    async def apost_vlm_as_llm(self, prompt: str) -> str:
        messages = self._build_vlm_message(prompt)
        payload = self._build_payload(messages)

        try:
            async with httpx.AsyncClient(timeout=100000.0) as client:
                resp = await client.post(self.url, headers=self.headers, json=payload)
                # resp.raise_for_status()
                return self._extract_response(resp.json())
        except Exception as e:
            logger.error(f"apost_vlm_as_llm 请求失败: {str(e)}")
            return str(e)

if __name__ == "__main__":
# CUDA_VISIBLE_DEVICES=0  VLLM_USE_V1=0 --api-key zjuici --served-model-name qwen2.5-pt 
# --port 23335 --uvicorn-log-level debug --dtype float16  --max-model-len 8192 --enable-log-requests  --gpu-memory-utilization 0.9
    url_chat = "https://qianfan.baidubce.com/v2/chat/completions"
    url_complete = "http://localhost:8000/v1/completions"
    llm_model_id = "qwen3-vl-8b-thinking"
    api_key = "bce-v3/ALTAK-vKka21ZWbdERPrRIu4Kvb/cdc77bce75384b8ff4b3d0666acd14e5f956ff22"
    clinet = LLMClient(url=url_chat, model_name=llm_model_id, api_key=api_key)
    res = clinet.post_llm("你是谁")
    print(res)