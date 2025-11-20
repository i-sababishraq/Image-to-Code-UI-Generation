# agent_azure_vlm.py
# .env exists in this folder, run ls -a to see.

"""
python agent_azure_vlm.py \
    --image "Images/2.png" \
    --out_rel_desc "Outputs/Agents/2/relative_desc.txt" \
    --out_brief "Outputs/Agents/2/gpt_ui_brief.txt" \
    --out_html "Outputs/Agents/2/reasoned_azure.html" \
    --vision_deployment "gpt-4.1-mini" \
    --text_deployment "gpt-4.1-mini" \
    --refine_max_iters 3 \
    --refine_threshold 8
"""
# ----------------------------------
# Converts the multi-stage pipeline into a LangGraph StateGraph
import os
import re
import json
import base64
import mimetypes
import argparse
import pathlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from uuid import uuid4
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

# ---------- Existing helpers (adapted) ----------

_SCORE_KEYS = ["aesthetics","completeness","layout_fidelity","text_legibility","visual_balance"]

def _section(text: str, name: str) -> str:
    pat = rf"{name}:\s*\n(.*?)(?=\n[A-Z_]+:\s*\n|\Z)"
    m = re.search(pat, text, flags=re.S)
    return m.group(1).strip() if m else ""

def _score_val(block: str, key: str, default: int = 0) -> int:
    m = re.search(rf"{key}\s*:\s*(-?\d+)", block, flags=re.I)
    try:
        return int(m.group(1)) if m else default
    except:
        return default
        
def encode_image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_html(text: str) -> str:
    m = re.search(r"```html(.*?)```", text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    i = text.lower().find("<html")
    return text[i:].strip() if i != -1 else text.strip()

def get_client(endpoint: str, api_key: str, api_version: str="2024-10-21") -> AzureOpenAI:
    if not api_key or not endpoint:
        raise RuntimeError("Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

def chat_complete(client: AzureOpenAI, deployment: str, messages: List[Dict[str, Any]],
                  temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

# ---------- Prompts (as in your original) ----------
RELDESC_SYSTEM = "You are a meticulous UI analyst who describes layouts as a single dense paragraph of relative relationships."
RELDESC_PROMPT = """
From the provided wireframe image, produce ONE detailed paragraph (no bullets, no lists, no headings, no JSON, no code)
that states the RELATIVE layout and styling of the page so a code generator can rebuild it.

Requirements for the paragraph:
- Mention the overall background color and the dominant text color (infer reasonable #HEX).
- Describe the NAV BAR first: its position (top row), left/center/right alignment of brand/logo and items, the item order,
  which (if any) is emphasized/active, approximate pill/underline treatment, and colors for default vs active.
- Describe the CONTENT in reading order by rows:
  * For each row, say HOW MANY items appear side-by-side and their approximate relative widths (e.g., equal thirds, 1/3–2/3).
  * For “cards” or boxes: state the title text (or short descriptor), the presence of body text, any CTA (button/link) labels,
    button shape (rounded/square), fill/outline, and inferred #HEX colors for surface, border, and CTA.
  * Call out approximate spacing (e.g., tight/medium/roomy), gutters/gaps (px if you can), and typical radii (px).
- Describe the FOOTER last: alignment (center/left/right), text size relative to body, and background/text colors.
- Include one sentence on typography: font family category (system sans/serif), approximate base size (px), and headings hierarchy.
- Keep everything in ONE paragraph. Do not use line breaks except normal wrapping.

Return ONLY that single paragraph.
"""

BRIEF_SYSTEM = "You are a senior product designer who converts wireframes into precise UI design briefs."
BRIEF_PROMPT = """
Using the RELATIVE LAYOUT DESCRIPTION (authoritative for relative structure) and the wireframe image,
write a **UI DESIGN BRIEF** using EXACTLY these section headings, in this order. Be concise but specific. Infer reasonable hex colors.
If there is any conflict, prefer the wireframe image but keep structure consistent with the relative description.

### UI_SUMMARY
One paragraph that states page purpose and the major regions.

### COLOR_PALETTE
List 6–10 colors as `name: #HEX` including background, surface/card, text, muted text, primary,
secondary/accent, link, button-default, button-active.

### TYPOGRAPHY
Font family (system stack), base font-size (px), title sizes (h1/h2/h3 in px), and weight rules.
Line-heights.

### LAYOUT
Container max-width (px), global padding (px), section gaps (px), and the overall structure
(header/nav, content rows/columns, footer). State **how many items appear side-by-side** in each row
and at which breakpoint they stack.

### NAVBAR
Exact order of items, which one is ACTIVE, and pill styling (padding, radius, default vs active
background/text colors).

### CARDS
For each card in the content row: title text, body summary, CTA label and style (button/link),
card padding, radius, shadow, spacing between title/body/cta.

### RESPONSIVE_RULES
Breakpoints (sm/md/lg in px) and what changes at each (column stack rules, spacing adjustments).

### SPACING_AND_BORDERS
Numbers (px) for margins, gaps, radii used across elements.

Output ONLY the brief text with those headings (no code fences, no JSON).
"""

CODE_SYSTEM = "You are a meticulous frontend engineer who writes clean, modern, responsive HTML+CSS."
CODE_PROMPT = """
Using the following **RELATIVE LAYOUT DESCRIPTION** and **UI DESIGN BRIEF**, generate a SINGLE, self-contained HTML document:

Requirements:
- Semantic tags: header/nav/main/section/footer.
- One <style> block; no external CSS/JS.
- Define CSS variables from the palette and use them consistently.
- Navbar items are pills; the 'active' nav uses the active button colors and stands out.
- Implement the layout: container max-width, gaps, grid columns, and stacking rules per breakpoints.
- Cards: match titles, body copy, CTA (as <a> styled like a button), padding, radius, and shadow.
- Links use the palette's link color; hover states are subtle (opacity or underline).
- Footer centered.
- Output ONLY valid HTML starting with <html> and ending with </html>.

RELATIVE LAYOUT DESCRIPTION:
"""

# PLAIN-TEXT JUDGE (no JSON)
SCORING_RUBRIC = r"""
You are an experienced front-end engineer. Compare two images: (A) the original wireframe, and (B) the generated HTML rendering,
and read the HTML/CSS code used for (B).

Return a PLAIN-TEXT report with the following sections EXACTLY in this order
(no JSON, no code fences around the whole report):

SCORES:
aesthetics: <0-10>
completeness: <0-10>
layout_fidelity: <0-10>   # be harsh; row alignment, relative sizes and positions must match A
text_legibility: <0-10>
visual_balance: <0-10>
aggregate: <float>        # mean of the five scores

ISSUES_TOP3:
- <short, specific issue 1>
- <issue 2>
- <issue 3>

LAYOUT_DIFFS:
- component: <nav|grid|card[1]|card[2]|footer>
  a_bbox_pct: [x,y,w,h]   # approx percentages (0–100) of page width/height in A
  b_bbox_pct: [x,y,w,h]   # same for B
  fix: <one sentence with exact px/cols/gaps>

CSS_PATCH:
```css
/* <= 40 lines, use existing selectors where possible; use px and hex colors */
.selector { property: value; }
/* ... */
```

HTML_EDITS:
- <one edit per line; selector + action, e.g., "add-class .card --class=wide":
- <allowed actions: move-before, move-after, insert-before, insert-after, set-attr, replace-text, add-class, remove-class>

REGENERATE_PROMPT:
<1–4 lines with exact grid, gaps (px), radii (px), hex colors, and font sizes to rebuild if needed>

FEEDBACK:
<one dense paragraph prioritizing layout_fidelity with exact px/cols/gaps/hex values>
"""

REFINE_SYSTEM = (
    "You are a senior frontend engineer who strictly applies critique to improve HTML/CSS while matching the wireframe."
)

REFINE_PROMPT = """
You are given:
1) (A) the original wireframe image
2) The CURRENT HTML (single-file) that produced (B) the rendering
3) A critique ("feedback") produced by a rubric-based comparison of A vs B

Task:
- Produce a NEW single-file HTML that addresses EVERY feedback point while staying faithful to A.
- Fix layout fidelity (columns, spacing, alignment), completeness (ensure all components in A exist),
  typography/contrast for legibility, and overall aesthetics and balance.
- Keep it self-contained (inline <style>; no external CSS/JS).
- Output ONLY valid HTML starting with <html> and ending with </html>.
"""

# ---------- State Schema ----------
@dataclass
class CodeRefineState:
    image_path: str
    out_rel_desc: str
    out_brief: str
    out_html: str
    vision_deployment: str
    text_deployment: str
    reldesc_tokens: int
    brief_tokens: int
    code_tokens: int
    judge_tokens: int
    temp: float
    refine_max_iters: int
    refine_threshold: int
    shot_width: int
    shot_height: int
    
    # Runtime state
    image_data_url: Optional[str] = None
    rel_desc: Optional[str] = None
    brief: Optional[str] = None
    html: Optional[str] = None
    current_iteration: int = 0
    scores: Optional[Dict[str, Any]] = None
    stop_refinement: bool = False
    
    messages: List[str] = field(default_factory=list)

# ---------- Pipeline Node Functions ----------
def node_stage0(state: CodeRefineState) -> CodeRefineState:
    # Stage 0: Image -> Relative Layout Description
    state.image_data_url = encode_image_to_data_url(state.image_path)
    client = get_client(os.getenv("ENDPOINT_URL", ""), os.getenv("AZURE_OPENAI_API_KEY", ""))
    messages = [
        {"role": "system", "content": RELDESC_SYSTEM},
        {"role": "user", "content": [
            {"type":"image_url", "image_url":{"url":state.image_data_url}},
            {"type":"text", "text": RELDESC_PROMPT.strip()},
        ]},
    ]
    state.rel_desc = chat_complete(client, state.vision_deployment, messages, state.temp, state.reldesc_tokens)
    state.messages.append("Stage0: Generated relative layout description.")
    # Save output
    pathlib.Path(state.out_rel_desc).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_rel_desc, "w", encoding="utf-8") as f:
        f.write(state.rel_desc.strip())
    state.messages.append(f"Saved rel_desc -> {state.out_rel_desc}")
    return state

def node_stage1(state: CodeRefineState) -> CodeRefineState:
    # Stage 1: Image + rel_desc -> UI Design Brief
    client = get_client(os.getenv("ENDPOINT_URL", ""), os.getenv("AZURE_OPENAI_API_KEY", ""))
    messages = [
        {"role": "system", "content": BRIEF_SYSTEM},
        {"role": "user", "content": [
            {"type":"image_url", "image_url":{"url":state.image_data_url}},
            {"type":"text", "text": BRIEF_PROMPT.strip() + "\n\nRELATIVE LAYOUT DESCRIPTION:\n" + state.rel_desc.strip()},
        ]},
    ]
    state.brief = chat_complete(client, state.vision_deployment, messages, state.temp, state.brief_tokens)
    state.messages.append("Stage1: Generated UI design brief.")
    pathlib.Path(state.out_brief).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_brief, "w", encoding="utf-8") as f:
        f.write(state.brief)
    state.messages.append(f"Saved brief -> {state.out_brief}")
    return state

def node_stage2(state: CodeRefineState) -> CodeRefineState:
    # Stage 2: Brief + rel_desc -> HTML
    client = get_client(os.getenv("ENDPOINT_URL", ""), os.getenv("AZURE_OPENAI_API_KEY", ""))
    messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": [
            {"type":"text", "text": CODE_PROMPT.strip()},
            {"type":"text", "text": "RELATIVE LAYOUT DESCRIPTION:\n" + state.rel_desc.strip()},
            {"type":"text", "text": "UI DESIGN BRIEF:\n" + state.brief.strip()},
        ]},
    ]
    raw = chat_complete(client, state.text_deployment, messages, state.temp, state.code_tokens)
    state.html = extract_html(raw)
    state.messages.append("Stage2: Generated HTML.")
    pathlib.Path(state.out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_html, "w", encoding="utf-8") as f:
        f.write(state.html)
    state.messages.append(f"Saved HTML -> {state.out_html}")
    return state

def node_stage3_score(state: CodeRefineState) -> CodeRefineState:
    # Stage 3: Render & Score
    # Use your existing render_html_to_png function (import or define)
    from playwright.sync_api import sync_playwright

    html_path = pathlib.Path(state.out_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(state.html)
    shot_png = html_path.with_name(html_path.stem + f".png")
    shot_png_path = shot_png
    # render current html to PNG
    shot_png.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": state.shot_width, "height": state.shot_height}, device_scale_factor=2.0)
        page = ctx.new_page()
        page.goto(pathlib.Path(state.out_html).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.screenshot(path=shot_png_path, full_page=True)
        ctx.close()
        browser.close()
    state.messages.append(f"Rendered HTML to PNG -> {shot_png_path}")
    
    # Score via LLM
    data_a = encode_image_to_data_url(state.image_path)
    data_b = encode_image_to_data_url(shot_png_path)
    client = get_client(os.getenv("ENDPOINT_URL", ""), os.getenv("AZURE_OPENAI_API_KEY", ""))
    messages = [
        {"role": "system", "content": "Return the specified PLAIN-TEXT report exactly as instructed."},
        {"role": "user", "content": [
            {"type": "text", "text": SCORING_RUBRIC.strip()},
            {"type": "text", "text": "Image (A): original wireframe"},
            {"type": "image_url", "image_url":{"url": data_a}},
            {"type": "text", "text": "Image (B): generated HTML rendering"},
            {"type": "image_url", "image_url":{"url": data_b}},
            {"type": "text", "text": "HTML/CSS code used to produce image (B):\n" + state.html}
        ]},
    ]
    resp = chat_complete(client, state.vision_deployment, messages, 0.0, state.judge_tokens)
    # parse scores (you’d reuse your parse_text_report logic)
    # For brevity assume parse_text_report exists
    # from azure_code_refinement4 import parse_text_report  # if you keep that function
    state.scores = parse_text_report(resp)
    state.messages.append(f"Stage3: Scoring done: {state.scores}")
    
    # Determine whether to stop refinement
    min_score = min(int(state.scores["scores"][k]) for k in ["aesthetics","completeness","layout_fidelity","text_legibility","visual_balance"])
    if min_score >= state.refine_threshold:
        state.stop_refinement = True
    return state

def parse_text_report(report: str) -> Dict[str, Any]:
    sb = _section(report, "SCORES")
    scores = {k: _score_val(sb, k, 0) for k in _SCORE_KEYS}
    m_agg = re.search(r"aggregate\s*:\s*([0-9]+(?:\.[0-9]+)?)", sb, flags=re.I)
    aggregate = float(m_agg.group(1)) if m_agg else sum(scores.values())/5.0

    css_patch = ""
    css_match = re.search(r"CSS_PATCH:\s*```css\s+(.*?)\s+```", report, flags=re.S|re.I)
    if css_match:
        css_patch = css_match.group(1).strip()

    html_edits = _section(report, "HTML_EDITS")
    regenerate_prompt = _section(report, "REGENERATE_PROMPT")
    feedback = _section(report, "FEEDBACK")
    issues = _section(report, "ISSUES_TOP3")
    layout_diffs = _section(report, "LAYOUT_DIFFS")

    return {
        "scores": scores,
        "aggregate": aggregate,
        "css_patch": css_patch,
        "html_edits": html_edits,
        "regenerate_prompt": regenerate_prompt,
        "feedback": feedback,
        "issues_top3": issues,
        "layout_diffs": layout_diffs,
        "raw": report,
    }
    
def refine_with_feedback(client: AzureOpenAI, vision_deployment: str, wireframe_image: str,
                         current_html: str, feedback: str,
                         css_patch: str = "", html_edits: str = "", regenerate_prompt: str = "",
                         temperature: float = 0.12, max_tokens: int = 2200) -> str:
    data_a = encode_image_to_data_url(wireframe_image)

    refine_instructions = f"""
{REFINE_PROMPT.strip()}

APPLY THESE PATCHES EXACTLY:

1) CSS PATCH (paste/replace inside <style>):
{css_patch or "/* none */"}

2) HTML EDITS (perform sequentially on current DOM; one per line):
{html_edits or "(none)"}

3) REGENERATE PROMPT (if structural rebuild needed, follow strictly):
{regenerate_prompt or "(none)"}

IMPORTANT:
- Keep single-file with inline <style>.
- Use existing class names/selectors; do not rename unless instructed.
- If a new rule conflicts, REPLACE existing rule with the patched value.
- Desktop layout must match the wireframe: respect exact columns/gaps/px values above.
"""

    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_a}},
            {"type": "text", "text": refine_instructions + "\n\nCURRENT_HTML:\n```html\n" + current_html + "\n```"}
        ]},
    ]
    out = chat_complete(client, vision_deployment, messages, temperature, max_tokens)
    html = extract_html(out)
    if "<html" not in html.lower():
        html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'><title>Refined</title></head>\n<body>\n{html}\n</body>\n</html>"
    return html
    
def node_refine_loop(state: CodeRefineState) -> CodeRefineState:
    # Refinement loop: if stop_refinement is False and iteration < max, then refine
    if state.stop_refinement or state.current_iteration >= state.refine_max_iters:
        state.messages.append("Refinement loop ended.")
        return state
    
    state.current_iteration += 1
    # Use your refine_with_feedback helper
    # from azure_code_refinement4 import refine_with_feedback  # if you keep that in module
    state.html = refine_with_feedback(
        client=get_client(os.getenv("ENDPOINT_URL", ""), os.getenv("AZURE_OPENAI_API_KEY", "")),
        vision_deployment=state.vision_deployment,
        wireframe_image=state.image_path,
        current_html=state.html,
        feedback=state.scores.get("feedback",""),
        css_patch=state.scores.get("css_patch",""),
        html_edits=state.scores.get("html_edits",""),
        regenerate_prompt=state.scores.get("regenerate_prompt",""),
        temperature=state.temp,
        max_tokens=state.code_tokens
    )
    # Save the new version of HTML to a versioned file
    versioned_path = pathlib.Path(state.out_html).with_name(pathlib.Path(state.out_html).stem + f"_v{state.current_iteration}" + pathlib.Path(state.out_html).suffix)
    with open(versioned_path, "w", encoding="utf-8") as f:
        f.write(state.html)

    state.out_html = str(versioned_path)
    state.messages.append(f"Saved refined HTML v{state.current_iteration} -> {versioned_path}")
    
    # Then we go back to scoring node
    return state

# ---------- Build the LangGraph workflow ----------
workflow = StateGraph(CodeRefineState)
workflow.add_node("stage0", node_stage0)
workflow.add_node("stage1", node_stage1)
workflow.add_node("stage2", node_stage2)
workflow.add_node("stage3_score", node_stage3_score)
workflow.add_node("refine_loop", node_refine_loop)

workflow.set_entry_point("stage0")

workflow.add_edge("stage0", "stage1")
workflow.add_edge("stage1", "stage2")
workflow.add_edge("stage2", "stage3_score")
workflow.add_edge("stage3_score", "refine_loop")

def decide_next(state: CodeRefineState) -> str:
    if not state.stop_refinement and state.current_iteration < state.refine_max_iters:
        return "refine_loop"
    # return END
    return "end"

workflow.add_conditional_edges("refine_loop", decide_next, {"refine_loop": "stage3_score", "end": END})

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ---------- CLI Runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_rel_desc", default="Outputs/relative_desc.txt")
    ap.add_argument("--out_brief", default="Outputs/ui_brief.txt")
    ap.add_argument("--out_html", default="Outputs/generated_page.html")
    ap.add_argument("--vision_deployment", required=True)
    ap.add_argument("--text_deployment", required=True)
    ap.add_argument("--reldesc_tokens", type=int, default=700)
    ap.add_argument("--brief_tokens", type=int, default=1100)
    ap.add_argument("--code_tokens", type=int, default=2200)
    ap.add_argument("--judge_tokens", type=int, default=900)
    ap.add_argument("--temp", type=float, default=0.12)
    ap.add_argument("--refine_max_iters", type=int, default=3)
    ap.add_argument("--refine_threshold", type=int, default=8)
    ap.add_argument("--shot_width", type=int, default=1536)
    ap.add_argument("--shot_height", type=int, default=900)
    args = ap.parse_args()

    state = CodeRefineState(
        image_path=args.image,
        out_rel_desc=args.out_rel_desc,
        out_brief=args.out_brief,
        out_html=args.out_html,
        vision_deployment=args.vision_deployment,
        text_deployment=args.text_deployment,
        reldesc_tokens=args.reldesc_tokens,
        brief_tokens=args.brief_tokens,
        code_tokens=args.code_tokens,
        judge_tokens=args.judge_tokens,
        temp=args.temp,
        refine_max_iters=args.refine_max_iters,
        refine_threshold=args.refine_threshold,
        shot_width=args.shot_width,
        shot_height=args.shot_height
    )

    run_id = f"wireframe-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    result = app.invoke(state, config=config)
    
    # result = app.invoke(state, config={"configurable":{"thread_id": f"run-{os.getpid()}"}})

    # Final saving (if needed) already done in nodes
    print("\n--- PIPELINE MESSAGES ---")
    print("\n".join(result["messages"]))
    print(f"\nFinal HTML saved to: {state.out_html}")

if __name__ == "__main__":
    main()
