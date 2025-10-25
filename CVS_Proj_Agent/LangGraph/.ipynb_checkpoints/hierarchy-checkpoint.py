#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import pandas as pd
from PIL import Image  # <-- NEW
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)

# --------------------------
# Defaults
# --------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_DIR = "Images"
DEFAULT_OUT_JSON_DIR = "outputs_hierarchy_json_25"
DEFAULT_OUT_CSV = "qwen25vl_component_hierarchy.csv"
DEFAULT_MAX_NEW_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

PROMPT_HIERARCHY = (
    "You are an expert UI layout analyzer. "
    "Analyze this wireframe and output all visible components in a hierarchical JSON structure.\n\n"
    "Each component should be represented as an object with:\n"
    "- 'type': the component name (e.g., header, nav, hero, button, image, card, footer)\n"
    "- 'attributes': a dictionary with attributes like color, position, size, alignment, and text content if visible\n"
    "- 'children': a list of nested components inside it\n\n"
    "The root node should represent the full page as 'page'.\n"
    "Follow the visual hierarchy (top to bottom, left to right). Output valid JSON only—no text outside the JSON."
)

def log(msg: str) -> None:
    print(msg, flush=True)

def find_json_block(text: str) -> Optional[str]:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i+1].strip()
    return None

def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_model_and_processor(model_name: str) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    has_cuda = torch.cuda.is_available()
    device_map = "auto" if has_cuda else {"": "cpu"}
    torch_dtype = torch.bfloat16 if has_cuda else torch.float32

    log(f"→ Loading model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def build_messages(pil_image: Image.Image, prompt: str) -> Dict[str, Any]:
    # The image is embedded in the chat turn; we will also pass it again via `images=[...]`
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt},
        ],
    }]

def open_pil_image(path: str) -> Image.Image:
    """
    Open image with Pillow, normalize to RGB, and use the first frame for animated formats.
    """
    img = Image.open(path)
    try:
        # If animated (GIF/WebP), take the first frame
        if getattr(img, "is_animated", False):
            img.seek(0)
    except Exception:
        pass
    # Convert to RGB to avoid mode issues (e.g., P/LA/CMYK)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def run_inference_on_image(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    force_json: bool = True,
) -> str:

    pil_img = open_pil_image(image_path)  # <-- OPEN AS PIL IMAGE
    messages = build_messages(pil_img, PROMPT_HIERARCHY)

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # IMPORTANT: pass the PIL image, not the string path
    inputs = processor(
        text=[chat_text],
        images=[pil_img],                  # <-- PIL image, not path
        return_tensors="pt"
    ).to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0.0),
        repetition_penalty=1.0,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, generation_config=gen_cfg)

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return decoded.strip() if force_json else decoded

def process_directory(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_dir: Path,
    out_json_dir: Path,
    out_csv_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    image_dir = image_dir.resolve()
    out_json_dir = out_json_dir.resolve()
    out_csv_path = out_csv_path.resolve()
    out_json_dir.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in exts]

    if not images:
        log(f"⚠️  No images found in: {image_dir}")
        return

    rows = []
    for img_path in images:
        log(f"\n🔍 Processing: {img_path.name}")
        row = {"image": img_path.name, "status": "", "raw_output_path": "", "json_path": ""}

        try:
            raw_text = run_inference_on_image(
                model=model,
                processor=processor,
                image_path=str(img_path),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                force_json=True,
            )

            raw_path = out_json_dir / f"{img_path.stem}_raw.txt"
            save_text(raw_path, raw_text)
            row["raw_output_path"] = str(raw_path)

            try:
                parsed_obj = json.loads(raw_text)
                json_path = out_json_dir / f"{img_path.stem}_hierarchy.json"
                save_json(json_path, parsed_obj)
                row["json_path"] = str(json_path)
                row["status"] = "parsed"
                log("✅ Parsed valid JSON directly.")
            except json.JSONDecodeError:
                candidate = find_json_block(raw_text)
                if candidate:
                    try:
                        parsed_obj = json.loads(candidate)
                        json_path = out_json_dir / f"{img_path.stem}_hierarchy.json"
                        save_json(json_path, parsed_obj)
                        row["json_path"] = str(json_path)
                        row["status"] = "parsed_from_block"
                        log("✅ Extracted and parsed JSON from within text.")
                    except json.JSONDecodeError as e:
                        clipped_path = out_json_dir / f"{img_path.stem}_candidate.json"
                        save_text(clipped_path, candidate)
                        row["status"] = f"invalid_json_candidate: {e}"
                        log(f"⚠️  Found JSON-like block but parse failed: {e}")
                else:
                    row["status"] = "invalid_json"
                    log("⚠️  No JSON object found in model output.")

        except Exception as e:
            row["status"] = f"error: {e}"
            log(f"❌ Error on {img_path.name}: {e}")

        rows.append(row)

    df = pd.DataFrame(rows, columns=["image", "status", "raw_output_path", "json_path"])
    df.to_csv(out_csv_path, index=False)
    log(f"\n🏁 Done. Summary: {out_csv_path}\n📂 JSON/text outputs: {out_json_dir}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract UI component hierarchy JSON from wireframe images using Qwen2.5-VL.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory with input images")
    ap.add_argument("--out-json-dir", default=DEFAULT_OUT_JSON_DIR, help="Directory to save per-image JSON/TXT outputs")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="CSV path for the run summary")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (0 for greedy)")
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p for nucleus sampling")
    return ap.parse_args()

def main():
    args = parse_args()

    log("======== Config ========")
    log(f"Model:           {args.model}")
    log(f"Images:          {args.image_dir}")
    log(f"Out JSON dir:    {args.out_json_dir}")
    log(f"Out CSV:         {args.out_csv}")
    log(f"Max new tokens:  {args.max_new_tokens}")
    log(f"Temperature:     {args.temperature}")
    log(f"Top-p:           {args.top_p}")
    log("========================")

    model, processor = load_model_and_processor(args.model)

    process_directory(
        model=model,
        processor=processor,
        image_dir=Path(args.image_dir),
        out_json_dir=Path(args.out_json_dir),
        out_csv_path=Path(args.out_csv),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

if __name__ == "__main__":
    main()
