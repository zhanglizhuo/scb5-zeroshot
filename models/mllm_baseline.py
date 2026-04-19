"""
models/mllm_baseline.py
多模态大语言模型零样本基线
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
支持：
  - LLaVA-1.6-Mistral-7B (int4量化，V100可跑)
  - Qwen-VL-Chat (int8量化)
  - GPT-4V (API调用)

推理策略：
  Chain-of-thought + 结构化输出（JSON格式）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import re
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from urllib import request
import numpy as np
from PIL import Image
import torch


# ─── 提示模板 ────────────────────────────────────────────────────────────────

MLLM_SYSTEM_PROMPT = """You are an expert in classroom activity recognition. 
You will be shown a classroom image and asked to identify the activity.
Always respond with valid JSON only."""


def build_classification_prompt(classes: List[str], multilabel: bool = False) -> str:
    """构建分类提示"""
    class_list = "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)])
    if multilabel:
        prompt = f"""Analyze this classroom image carefully.

Available categories (multiple can apply):
{class_list}

Instructions:
1. Think step by step about what activities are visible
2. A teacher or student may exhibit multiple behaviors simultaneously
3. Return a JSON with your analysis

Respond ONLY with this JSON format:
{{
  "reasoning": "brief description of what you observe",
  "predicted_labels": [list of integer indices that apply],
  "confidence": "high/medium/low"
}}"""
    else:
        prompt = f"""Analyze this classroom image carefully.

Choose the SINGLE best matching category:
{class_list}

Instructions:
1. Think step by step about the main activity visible
2. Select only one category that best describes the scene
3. Return a JSON with your analysis

Respond ONLY with this JSON format:
{{
  "reasoning": "brief description of what you observe",
  "predicted_label": integer index of the best matching category,
  "confidence": "high/medium/low"
}}"""
    return prompt


def build_chain_of_thought_prompt(classes: List[str]) -> str:
    """带链式推理的提示"""
    class_list = "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)])
    return f"""You are analyzing a classroom image for behavior recognition.

Step 1: Describe what you see (people, actions, objects)
Step 2: Identify key behavioral cues
Step 3: Match to the closest category

Categories:
{class_list}

Respond in JSON:
{{
  "observation": "what you see in the image",
  "key_cues": ["list", "of", "visual", "cues"],
  "reasoning": "why you chose this category",
  "predicted_label": integer,
  "predicted_labels": [list of integers if multiple apply]
}}"""


# ─── LLaVA 推理器 ────────────────────────────────────────────────────────────

class LLaVAInferencer:
    """
    LLaVA-1.6-Mistral-7B 零样本推理器
    int4量化后约需 10GB VRAM，V100 32GB 可运行
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda:0",
        quantization: str = "int4",
        max_new_tokens: int = 256,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = self._load_model(model_id, quantization)
        print(f"[LLaVA] Loaded on {device}")

    def _load_model(self, model_id: str, quantization: str):
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
        )
        processor = LlavaNextProcessor.from_pretrained(model_id)

        if quantization == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=self.device,
            )
        elif quantization == "int8":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map=self.device,
            )
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        return model, processor

    def _parse_json_response(self, response: str, multilabel: bool) -> Dict:
        """解析 JSON 格式响应"""
        # 提取 JSON 块
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed
            except json.JSONDecodeError:
                pass

        # 回退：用正则提取数字
        if multilabel:
            nums = re.findall(r'\d+', response)
            return {"predicted_labels": [int(n) for n in nums[:5]]}
        else:
            nums = re.findall(r'\d+', response)
            return {"predicted_label": int(nums[0]) if nums else 0}

    def infer_batch(
        self,
        images: List[Image.Image],
        classes: List[str],
        multilabel: bool = False,
        use_cot: bool = True,
    ) -> List[Dict]:
        """批量推理（每次一张，因为 MLLM 不支持真正的图像批处理）"""
        prompt_text = (build_chain_of_thought_prompt(classes) if use_cot
                       else build_classification_prompt(classes, multilabel))

        results = []
        for img in images:
            # 构建对话格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            inputs = self.processor(
                images=img, text=prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            response = self.processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            parsed = self._parse_json_response(response, multilabel)
            results.append(parsed)

        return results

    def infer_dataset(
        self,
        dataloader,
        classes: List[str],
        multilabel: bool = False,
        num_classes: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对整个数据集推理，返回 (logits/preds, labels)
        对于 MLLM，我们返回 one-hot 预测而不是 softmax 分数
        """
        num_classes = num_classes or len(classes)
        all_preds = []
        all_labels = []

        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="LLaVA inference"):
            # 从 tensor 还原 PIL Image（近似，实际应在 dataset 端保存原始图像）
            images = batch["image"]
            labels = batch["label"].numpy()

            for i in range(len(images)):
                # 简单反归一化用于展示（实际应保存原始 PIL）
                img_np = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                result = self.infer_batch([pil_img], classes, multilabel)[0]

                pred = np.zeros(num_classes, dtype=float)
                if multilabel:
                    for idx in result.get("predicted_labels", []):
                        if 0 <= idx < num_classes:
                            pred[idx] = 1.0
                else:
                    idx = result.get("predicted_label", 0)
                    if 0 <= idx < num_classes:
                        pred[idx] = 1.0

                all_preds.append(pred)
                all_labels.append(labels[i])

        return np.array(all_preds), np.array(all_labels)


# ─── Qwen-VL 推理器 ───────────────────────────────────────────────────────────

class QwenVLInferencer:
    """
    Qwen-VL-Chat 零样本推理器
    int8量化后约需 14GB VRAM，V100 32GB 可运行
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-VL-Chat",
        device: str = "cuda:1",
        quantization: str = "int8",
        max_new_tokens: int = 256,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load_model(model_id, quantization)
        print(f"[QwenVL] Loaded on {device}")

    def _load_model(self, model_id: str, quantization: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if quantization == "int8":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
        model.eval()
        return model, tokenizer

    def _save_temp_image(self, img: Image.Image, path: str = "/tmp/qwen_temp.jpg"):
        img.save(path, format="JPEG", quality=95)
        return path

    def infer_single(
        self,
        img: Image.Image,
        classes: List[str],
        multilabel: bool = False,
    ) -> Dict:
        img_path = self._save_temp_image(img)
        prompt = build_classification_prompt(classes, multilabel)

        query = self.tokenizer.from_list_format([
            {"image": img_path},
            {"text": prompt},
        ])
        response, _ = self.model.chat(
            self.tokenizer,
            query=query,
            history=None,
            system=MLLM_SYSTEM_PROMPT,
        )

        # 解析响应
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass
        return {"predicted_label": 0, "predicted_labels": [0]}


# ─── GPT-4V 推理器（API版）──────────────────────────────────────────────────

class GPT4VInferencer:
    """
    GPT-4V API 调用（需要 OpenAI API Key）
    用于对比实验，提供上界参考
    """

    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def _encode_image(self, img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def infer_single(
        self,
        img: Image.Image,
        classes: List[str],
        multilabel: bool = False,
    ) -> Dict:
        b64_img = self._encode_image(img)
        prompt = build_chain_of_thought_prompt(classes)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": MLLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_img}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=512,
                temperature=0,
            )
            content = response.choices[0].message.content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"GPT-4V error: {e}")
        return {"predicted_label": 0, "predicted_labels": [0]}


# ─── Ollama Vision 推理器（本地）──────────────────────────────────────────

class OllamaVisionInferencer:
    """
    本地 Ollama 视觉模型推理器。
    适配支持 vision 能力的模型（如 qwen3.5:9b, gemma4:e4b）。
    """

    def __init__(self, model_id: str, host: str = "http://127.0.0.1:11434", timeout: int = 120):
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.timeout = timeout
        print(f"[OllamaVision] model={model_id} host={self.host}")

    def _encode_image(self, img: Image.Image) -> str:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _post_chat(self, prompt: str, image_b64: str) -> str:
        payload = {
            "model": self.model_id,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": MLLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [image_b64]},
            ],
            "options": {"temperature": 0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        return obj.get("message", {}).get("content", "")

    def infer_single(self, img: Image.Image, classes: List[str], multilabel: bool = False) -> Dict:
        prompt = build_classification_prompt(classes, multilabel)
        b64_img = self._encode_image(img)

        try:
            response = self._post_chat(prompt, b64_img)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[OllamaVision] infer error ({self.model_id}): {e}")

        # fallback parsing
        nums = re.findall(r'\d+', response if 'response' in locals() else "")
        if multilabel:
            return {"predicted_labels": [int(n) for n in nums[:5]]}
        return {"predicted_label": int(nums[0]) if nums else 0}


# ─── 工具：从 logits 提取结构化预测 ─────────────────────────────────────────

def mllm_preds_to_logits(
    preds: List[Dict],
    num_classes: int,
    multilabel: bool = False,
) -> np.ndarray:
    """
    将 MLLM 输出的字典列表转换为 (N, C) 伪 logit 矩阵
    （用 one-hot 表示，与 CLIP 输出格式统一）
    """
    N = len(preds)
    logits = np.zeros((N, num_classes), dtype=float)
    for i, pred in enumerate(preds):
        if multilabel:
            for idx in pred.get("predicted_labels", []):
                if 0 <= idx < num_classes:
                    logits[i, idx] = 1.0
        else:
            idx = pred.get("predicted_label", 0)
            if 0 <= idx < num_classes:
                logits[i, idx] = 1.0
            # 如果有多标签字段也处理
            for idx2 in pred.get("predicted_labels", []):
                if 0 <= idx2 < num_classes:
                    logits[i, idx2] = max(logits[i, idx2], 0.5)
    return logits


if __name__ == "__main__":
    print("MLLM baseline module loaded successfully.")
    print("Note: Actual inference requires GPU and model weights.")
    print("GPU requirements:")
    print("  LLaVA-1.6-7B (int4):  ~10GB VRAM")
    print("  Qwen-VL-Chat (int8):  ~14GB VRAM")
    print("  GPT-4V:               API only")
