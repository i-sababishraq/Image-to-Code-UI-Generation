# agent_vlm.py
# Minimal agentic pipeline (2 nodes): Qwen2.5-VL hierarchy -> Qwen2.5-VL HTML/CSS (model-driven styling)
# pip install langgraph transformers accelerate pillow

import os, io, json, base64, argparse, pathlib
from uuid import uuid4
from typing import Any, Dict, List, Optional

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = 2400

PROMPT_HIERARCHY = (
    "You are an expert UI layout analyzer. "
    "Analyze this wireframe and output all visible components as a hierarchical JSON.\n\n"
    "Each node must have:\n"
    "  - 'type': page|header|nav|section|card|text|paragraph|image|button|footer\n"
    "  - 'attributes': dict with attributes like alignment/position/size and text if visible; "
    "    include bbox if available as {\"x\":int,\"y\":int,\"w\":int,\"h\":int}\n"
    "  - 'children': ordered list of nested components (top->bottom, left->right)\n\n"
    "The root node must be 'page'. Output valid JSON only — no extra text."
)

# -----------------------------
# Utils
# -----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        pass
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        try: return json.loads(s[first:last+1])
        except Exception: pass
    # bracket-match fallback
    stack, start = [], None
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i); start = i if start is None else start
        elif ch == "}":
            if stack: stack.pop()
            if not stack and start is not None:
                snip = s[start:i+1]
                try: return json.loads(snip)
                except Exception: start = None
    return None

# -----------------------------
# Qwen2.5-VL wrapper
# -----------------------------
class QwenVL:
    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True
        )

    def infer_hierarchy(self, image_path: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_path], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        parsed = try_parse_json(out)
        if not parsed:
            return {"type":"page","attributes":{},"children":[],"_raw": out}
        return parsed.get("page", parsed)

    def synthesize_code(self, hierarchy: Dict[str, Any], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        Ask Qwen to generate FULL HTML+CSS purely from the hierarchy.
        No hardcoded CSS. The model infers styles/layout from types/attributes/bboxes.
        """
        system = {
            "role": "system",
            "content": [
                {"type": "text", "text":
                    "You convert UI hierarchies into complete, production-ready HTML + CSS. "
                    "Preserve visual structure and sequencing (top->bottom, left->right). "
                    "Honor alignment, grouping, and any size/position hints in attributes/bbox. "
                    "Use semantic HTML where appropriate. Output ONLY the final HTML document."
                }
            ]
        }
        user = {
            "role": "user",
            "content": [
                {"type": "text", "text":
                    "Given this UI hierarchy JSON, generate a single self-contained HTML file with a <style> block.\n"
                    "Requirements:\n"
                    "- Infer layout (rows/columns/stacking) and styling from hierarchy attributes and types.\n"
                    "- Keep colors/typography consistent and minimal; do not reference external assets.\n"
                    "- Include all components (header/nav/cards/text/buttons/images/footer) in the correct order.\n"
                    "- Use CSS Grid or Flexbox as needed; add simple responsive rules if appropriate.\n"
                    "- Return ONLY valid HTML (no explanations)."
                },
                {"type": "text", "text": json.dumps(hierarchy, ensure_ascii=False, indent=2)}
            ]
        }

        # text-only run
        text = self.processor.apply_chat_template([system, user], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=None, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        html = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Try to trim to the HTML doc if the model echoed the prompt
        start = html.find("<!DOCTYPE")
        if start == -1:
            start = html.find("<html")
        end = html.rfind("</html>")
        if start != -1 and end != -1:
            html = html[start:end+7]
        return html.strip()

# Single instance
qwen_vl = QwenVL()

# -----------------------------
# Node A — analyze (image -> hierarchy JSON, in memory)
# -----------------------------
def node_analyze(state: Dict[str, Any]) -> Dict[str, Any]:
    img_path = state["image_path"]
    img = load_image(img_path)
    hierarchy = qwen_vl.infer_hierarchy(img_path, PROMPT_HIERARCHY)
    msgs = state.get("messages", [])
    msgs.append("analyze: hierarchy extracted")
    return {"hierarchy": hierarchy, "image_b64": pil_to_b64(img), "messages": msgs}

# -----------------------------
# Node B — synthesize (hierarchy -> HTML+CSS via Qwen)
# -----------------------------
def node_synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    hierarchy = state.get("hierarchy", {"type":"page","children":[]})
    html = qwen_vl.synthesize_code(hierarchy)
    msgs = state.get("messages", [])
    msgs.append("synthesize: HTML/CSS generated by model")
    return {"html_css": html, "messages": msgs}

# -----------------------------
# Build graph (2 nodes)
# -----------------------------
workflow = StateGraph(dict)
workflow.add_node("analyze", node_analyze)
workflow.add_node("synthesize", node_synthesize)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to wireframe PNG/JPG")
    p.add_argument("--out_html", default="generated.html")
    args = p.parse_args()

    init_state = {"image_path": str(pathlib.Path(args.image).resolve()), "messages": []}
    config = {"configurable": {"thread_id": f"run-{uuid4()}"}}

    final: Dict[str, Any] = app.invoke(init_state, config=config)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(final.get("html_css", "<!-- empty -->"))

    print("\n--- PIPELINE MESSAGES ---")
    print("\n".join(final.get("messages", [])))
    print(f"\nSaved: {args.out_html}")
