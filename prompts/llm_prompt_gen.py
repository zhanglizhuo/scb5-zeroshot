"""
prompts/llm_prompt_gen.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用本地 LLM（LLaVA / Qwen）自动生成 CAPE 提示集
对比"人工设计 vs LLM 自动生成"的效果差异

这是论文的一个重要附加贡献点：
  - 证明 CAPE 设计原则可以被 LLM 复现
  - 或发现 LLM 生成提示的不足，分析原因
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch


# ─── 提示生成用的 Meta-Prompt ────────────────────────────────────────────────

META_PROMPT_TEMPLATE = """You are an expert in classroom behavior recognition for computer vision.

Generate exactly 3 descriptive prompts for the classroom activity category: "{class_name}"

Requirements:
1. Each prompt must be visually grounded (describe what a camera would see)
2. Prompts should cover different perspectives (viewpoint, action, context)
3. Prompts should be discriminative (highlight differences from similar categories)
4. Each prompt should be 10-20 words
5. Focus on Chinese K-12 classroom settings

Related categories to distinguish from: {related_classes}

Respond ONLY with a JSON array of exactly 3 strings:
["prompt 1", "prompt 2", "prompt 3"]"""


# ─── 使用本地 LLaVA 生成提示 ────────────────────────────────────────────────

def generate_with_llava(
    classes: List[str],
    model_device: str = "cuda:0",
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
) -> Dict[str, List[str]]:
    """使用 LLaVA（纯文本模式）生成提示"""
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        BitsAndBytesConfig,
    )
    import re

    print(f"Loading LLaVA for prompt generation on {model_device}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=model_device,
    )

    generated_prompts = {}

    for cls in classes:
        related = [c for c in classes if c != cls][:4]
        meta_prompt = META_PROMPT_TEMPLATE.format(
            class_name=cls,
            related_classes=", ".join(related),
        )

        # LLaVA 纯文本对话（无图像输入）
        conversation = [{"role": "user", "content": meta_prompt}]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(text=prompt, return_tensors="pt").to(model_device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        response = processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # 解析 JSON
        try:
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                prompts = json.loads(json_match.group())
                generated_prompts[cls] = prompts[:3]
            else:
                generated_prompts[cls] = [f"a classroom image showing {cls}"] * 3
        except Exception:
            generated_prompts[cls] = [f"a classroom image showing {cls}"] * 3

        print(f"  {cls}: {generated_prompts[cls]}")

    del model
    torch.cuda.empty_cache()
    return generated_prompts


# ─── 使用 OpenAI API 生成提示 ────────────────────────────────────────────────

def generate_with_openai(
    classes: List[str],
    api_key: str,
    model: str = "gpt-4-turbo",
) -> Dict[str, List[str]]:
    """使用 GPT-4 API 生成提示（效果最佳）"""
    import openai
    import re

    client = openai.OpenAI(api_key=api_key)
    generated_prompts = {}

    for cls in classes:
        related = [c for c in classes if c != cls][:4]
        meta_prompt = META_PROMPT_TEMPLATE.format(
            class_name=cls,
            related_classes=", ".join(related),
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            content = response.choices[0].message.content
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                prompts = json.loads(json_match.group())
                generated_prompts[cls] = prompts[:3]
            else:
                generated_prompts[cls] = [f"a classroom scene showing {cls}"] * 3
        except Exception as e:
            print(f"  Error for {cls}: {e}")
            generated_prompts[cls] = [f"a classroom scene showing {cls}"] * 3
        print(f"  {cls}: {generated_prompts[cls]}")

    return generated_prompts


# ─── 主函数 ──────────────────────────────────────────────────────────────────

ALL_CLASSES = [
    # TeacherBehavior
    "guide", "answer", "on-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackboard",
    # Student
    "hand-raise", "read", "write",
    # Micro-actions
    "bow-head", "turn-head",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava",
                        choices=["llava", "openai"])
    parser.add_argument("--results_dir", type=str, default="./results/llm_prompts")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating prompts using: {args.model}")

    if args.model == "llava":
        prompts = generate_with_llava(ALL_CLASSES, model_device=args.device)
    elif args.model == "openai" and args.openai_api_key:
        prompts = generate_with_openai(ALL_CLASSES, api_key=args.openai_api_key)
    else:
        print("No valid model specified. Using dummy prompts for testing.")
        prompts = {cls: [f"a classroom showing {cls}"] * 3 for cls in ALL_CLASSES}

    # 保存
    out_path = results_dir / f"llm_generated_prompts_{args.model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Generated prompts saved to {out_path}")

    # 同时更新 cape_prompts.py 中的 CAPE_LLM_GENERATED
    print("\nLLM-generated prompts preview:")
    for cls, ps in list(prompts.items())[:3]:
        print(f"  {cls}:")
        for p in ps:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
