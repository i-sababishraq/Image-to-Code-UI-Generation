#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage:
'''
python hierarchy_code.py \
  --image_dir Images \
  --raw_hierarchy_dir outputs_hierarchy_json_25 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --out_html_dir outputs_html_verify_25 \
  --outcsv qwen25vl_verify_hierarchy_results_from_rawtxt.csv
'''

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Any, Tuple, List

import torch
import pandas as pd
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)

# --------------------------
# Defaults / Config
# --------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_DIR = "Images"
DEFAULT_RAW_HIERARCHY_DIR = "outputs_hierarchy_json_25"
DEFAULT_OUTPUT_HTML_DIR = "outputs_html_verify_25"
DEFAULT_OUTPUT_CSV = "qwen25vl_verify_hierarchy_results_from_rawtxt.csv"
DEFAULT_MAX_NEW_TOKENS = 2500

PROMPT_HTML_FROM_JSON = (
    "You are an expert front-end developer. "
    "Use BOTH the following hierarchical component JSON and the provided wireframe image "
    "to generate a complete, minimal, responsive HTML5 layout.\n\n"
    "Guidelines:\n"
    "- Each node's 'type' corresponds to an HTML section or element.\n"
    "- Use semantic HTML tags (header, nav, main, section, article, footer, etc.).\n"
    "- Use node 'attributes' to infer inline styles (colors, alignment, size).\n"
    "- Preserve the hierarchy: parent nodes contain their children in proper order.\n"
    "- Include a minimal <style> block in <head> but no external CSS or JavaScript.\n"
    "- Use placeholder text for headings, paragraphs, or buttons.\n"
    "- The visual layout should reflect the image as closely as possible.\n\n"
    "Return ONLY valid HTML starting with <!doctype html>. No backticks, no explanations."
)

# --------------------------
# Utilities
# --------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def open_pil_image(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Extract a JSON object from a raw model output text.
    - Prefer fenced ```json blocks
    - Otherwise, find the first balanced {...} block
    """
    # Prefer fenced json block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Balanced-brace scan
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i+1].strip()
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Very mild repair: trim trailing commas
                        candidate2 = re.sub(r",(\s*[}\]])", r"\1", candidate)
                        try:
                            return json.loads(candidate2)
                        except Exception:
                            return None
    return None

def extract_html_only(text: str) -> str:
    """
    Strip code fences / chatter and return only the <!doctype html> ... </html> region if present.
    """
    # Remove code fences
    text = re.sub(r"^```.*?\n|```$", "", text, flags=re.DOTALL | re.MULTILINE).strip()

    # Try to locate a proper HTML document
    m = re.search(r"(?is)(<!doctype html.*?<html.*?>.*?</html>)", text)
    if m:
        return m.group(1).strip()

    # As fallback, if it starts with doctype, trust the whole text
    if text.lower().lstrip().startswith("<!doctype html"):
        return text.strip()

    # Last resort: try to wrap a found <html>..</html>
    m2 = re.search(r"(?is)(<html.*?>.*?</html>)", text)
    if m2:
        return "<!doctype html>\n" + m2.group(1).strip()

    # If nothing found, just return text; caller can still save for debugging
    return text.strip()

# --------------------------
# Model / Processor
# --------------------------
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

# --------------------------
# Inference
# --------------------------
def generate_html_from_json_and_image(
    json_data: Any,
    image_path: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:

    pil_img = open_pil_image(image_path)
    json_text = json.dumps(json_data, indent=2, ensure_ascii=False)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},  # pass PIL, not path
            {"type": "text", "text": PROMPT_HTML_FROM_JSON},
            {"type": "text", "text": f"Here is the JSON:\n\n{json_text}"},
        ],
    }]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[chat_text],
        images=[pil_img],     # pass PIL, not path
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, generation_config=gen_cfg)

    # Trim prompt tokens from the left (safer than decoding then splitting)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    # Ensure we only return valid HTML
    return extract_html_only(decoded)

# --------------------------
# Main pipeline
# --------------------------
def process_all(
    model_name: str,
    image_dir: Path,
    raw_hierarchy_dir: Path,
    out_html_dir: Path,
    out_csv_path: Path,
    max_new_tokens: int,
) -> None:

    model, processor = load_model_and_processor(model_name)

    image_dir = image_dir.resolve()
    raw_hierarchy_dir = raw_hierarchy_dir.resolve()
    out_html_dir = out_html_dir.resolve()
    out_csv_path = out_csv_path.resolve()

    out_html_dir.mkdir(parents=True, exist_ok=True)

    # Map image stems → available file
    def find_image_for_stem(stem: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    raw_files = sorted([p for p in raw_hierarchy_dir.iterdir() if p.name.endswith("_raw.txt")])

    if not raw_files:
        log(f"⚠️  No *_raw.txt files found in: {raw_hierarchy_dir}")
        return

    rows: List[dict] = []
    for raw_path in raw_files:
        stem = raw_path.stem.replace("_raw", "")
        image_path = find_image_for_stem(stem)

        if not image_path:
            log(f"⚠️  Skipping {raw_path.name} — no matching image found in {image_dir}.")
            rows.append({
                "raw_file": raw_path.name,
                "image_file": "",
                "status": "no_image",
                "html_file": "",
                "raw_output_excerpt": "",
            })
            continue

        log(f"\n🔍 Processing {raw_path.name} with image {image_path.name} ...")
        row = {
            "raw_file": raw_path.name,
            "image_file": image_path.name,
            "status": "",
            "html_file": "",
            "raw_output_excerpt": "",
        }

        try:
            raw_text = raw_path.read_text(encoding="utf-8")

            json_data = extract_json_from_text(raw_text)
            if json_data is None:
                row["status"] = "error: no valid JSON"
                rows.append(row)
                continue

            html_out = generate_html_from_json_and_image(
                json_data=json_data,
                image_path=str(image_path),
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
            )

            # Save HTML and log
            out_html_path = out_html_dir / f"{stem}__verify.html"
            save_text(out_html_path, html_out)
            row["html_file"] = str(out_html_path)
            row["status"] = "success"
            row["raw_output_excerpt"] = html_out[:400].replace("\n", " ")

        except Exception as e:
            row["status"] = f"error: {e}"

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv_path, index=False)
    log(f"\n✅ Verification done! HTMLs saved in {out_html_dir}")
    log(f"📊 Log saved to {out_csv_path}")

# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify Qwen2.5-VL hierarchies by generating HTML from JSON + image.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    ap.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory with input images")
    ap.add_argument("--raw_hierarchy_dir", default=DEFAULT_RAW_HIERARCHY_DIR,
                    help="Directory containing *_raw.txt files (hierarchy JSON text)")
    ap.add_argument("--out_html_dir", default=DEFAULT_OUTPUT_HTML_DIR, help="Directory to save generated HTML files")
    ap.add_argument("--out_csv", default=DEFAULT_OUTPUT_CSV, help="CSV path for the verification summary")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens to generate")
    return ap.parse_args()

def main():
    args = parse_args()

    log("======== Config ========")
    log(f"Model:             {args.model}")
    log(f"Images:            {args.image_dir}")
    log(f"Raw hierarchy dir: {args.raw_hierarchy_dir}")
    log(f"Out HTML dir:      {args.out_html_dir}")
    log(f"Out CSV:           {args.out_csv}")
    log(f"Max new tokens:    {args.max_new_tokens}")
    log("========================")

    process_all(
        model_name=args.model,
        image_dir=Path(args.image_dir),
        raw_hierarchy_dir=Path(args.raw_hierarchy_dir),
        out_html_dir=Path(args.out_html_dir),
        out_csv_path=Path(args.out_csv),
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()
