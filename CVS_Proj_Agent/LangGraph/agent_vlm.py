# wireframe_to_code_graph.py
import os, json, base64, io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from io import BytesIO

from PIL import Image
from uuid import uuid4

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# HF Transformers (Qwen2.5-VL)
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline, AutoModelForCausalLM, AutoTokenizer

# Asset Agent imports
import requests
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
#from langchain.output_parsers import RetryOutputParser
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
# Model names: choose an 8B/7B Qwen2.5-VL Instruct variant you have access to.
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")  # use 8B-equivalent if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Optional: pin transformers if you hit regressions (see notes in README).
# pip install "transformers==4.49.0"

# -----------------------------
# Utilities
# -----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def pil_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def b64img(pil_img: Image.Image) -> str:
    return base64.b64encode(pil_to_bytes(pil_img)).decode("utf-8")

def ensure_list(x):
    return x if isinstance(x, list) else [x]

# -----------------------------
# Qwen2.5-VL loader (single instance reused by nodes)
# -----------------------------
class QwenVL:
    def __init__(self, model_name: str = QWEN_VL_MODEL_NAME, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

    def chat(self, messages, temperature=0.2, max_new_tokens=2048):
        # messages must have typed content blocks as above
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)
    
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
    
        # slice off the prompt tokens before decoding (per HF example)
        gen_only = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        text = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return text


# Singleton (so we don’t reload per node)
qwen_vl = QwenVL()

# -----------------------------
# LangGraph State
# -----------------------------
@dataclass
class UI2CodeState:
    image_path: str
    image_b64: Optional[str] = None

    # Step outputs
    description_json: Optional[Dict[str, Any]] = None  # parsed components
    plan_json: Optional[Dict[str, Any]] = None         # ordered build plan
    html_css: Optional[str] = None                     # final HTML+CSS
    asset_paths: Dict[str, str] = field(default_factory=dict)  # path to found assets

    # Trace
    messages: List[str] = field(default_factory=list)

# -----------------------------
# Asset Agent Workflow State & Pydantic Models (from asset agent)
# -----------------------------
class AssetState(TypedDict):
    instructions: str
    bounding_box: Tuple[int, int]
    user_asset_options: Dict[str, str]
    search_successful: bool
    final_asset_path: Optional[str]
    route_decision: str
    error: Optional[str]

class RouteQuery(PydanticBaseModel):
    destination: str = PydanticField(description="Next node. Must be 'search' or 'use_user_asset'.")

# -----------------------------
# Node 1 — Perception: JSON UI description
# -----------------------------
def node_perception(state: UI2CodeState) -> UI2CodeState:
    img = load_image(state.image_path)
    print("image loaded")
    state.image_b64 = b64img(img)

    sys = {
        "role": "system",
        "content":[{"type": "text", "text": "You are a UI perception expert. Return precise JSON describing UI elements from a wireframe."}]
    }

    user = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Analyze this wireframe image and produce a STRICT JSON object describing its UI components. **You must generate unique component IDs and all other values based on the image.**"},
        {"type": "image", "image": img},
        {"type": "text", "text": """
Output a JSON object with this schema exactly. The values in the example are placeholders; you must replace them with values you derive from the image.

{
  "page": {
    "size": {"width": <image_width_in_pixels>, "height": <image_height_in_pixels>},
    "components": [
      {
        "id": "<unique_component_id_1>",
        "type": "<one_of_the_following: container|navbar|header|footer|section|div|grid|card|table|form|input|button|icon|image|text>",
        "role": "<a_semantic_role_like: logo|menu|hero|sidebar|content|cta|caption|paragraph|label>",
        "bbox": {"x": <x_coordinate>, "y": <y_coordinate>, "w": <width>, "h": <height>},
        "props": {"text": "<text_content_if_any>", "icon": "<icon_name_if_any>"},
        "children": ["<child_component_id_1>", "<child_component_id_2>"]
      }
    ]
  }
}
Only return JSON. No explanations.
"""
}]}


    resp = qwen_vl.chat([sys, user], temperature=0.1, max_new_tokens=1500)
    # Make it robust to stray text
    try:
        json_str = resp[resp.find("{"):resp.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"page": {"size": {}, "components": []}, "raw": resp}

    state.description_json = parsed
    state.messages.append("Perception complete: extracted JSON.")

    print("done describing the UI components in the image")
    return state

# -----------------------------
# Node 2 — Planner: order & hierarchy for code emission
# -----------------------------
def node_planner(state: UI2CodeState) -> UI2CodeState:
    assert state.description_json, "Missing description_json"

    sys = {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a front-end architect. Produce an ordered build plan from component JSON."}
        ],
    }
    user = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Given the component JSON, return a build plan. Analyze the input to determine the build order, hierarchy, visual style, and all assets needed."},
        {"type": "text", "text": json.dumps(state.description_json, ensure_ascii=False, indent=2)},
        {"type": "text", "text": """
Return a STRICT JSON object following this schema. 
**IMPORTANT**: The values in the example below are placeholders. You MUST populate the JSON with actual values derived from the component JSON provided above.

{
  "order": ["<component_id_1>", "<component_id_2>", ...],
  "regions": [
    {"name": "<semantic_region_name>", "components": ["<id_1>", "<id_2>"]}
  ],
  "hierarchy": [
    {"parent": "<parent_component_id>", "children": ["<child_id_1>", "<child_id_2>"]}
  ],
  "css_guidelines": {
    "colors": ["<primary_color>", "<secondary_color>", ...],
    "fonts": ["<main_font>", "system-ui"],
    "layout": "<description_of_layout_strategy>",
    "style": "<description_of_visual_style>"
  },
  "assets_needed": [
    {
      "component_id": "<id_of_an_image_component_from_input>",
      "description": "<a_detailed_description_of_what_the_image_should_be>",
      "bounding_box": {"width": "<actual_width_from_bbox>", "height": "<actual_height_from_bbox>"}
    },
         ...
  ]
}
If no assets are needed, return an empty list: "assets_needed": [].
Only return JSON.
"""}
    ],
}


    resp = qwen_vl.chat([sys, user], temperature=0.2, max_new_tokens=1200)
    try:
        json_str = resp[resp.find("{"):resp.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"order": [], "regions": [], "hierarchy": [], "css_guidelines": {}, "assets_needed": [], "raw": resp}

    state.plan_json = parsed
    state.messages.append("Planner complete: produced build order and guidelines.")
    return state

# -----------------------------
# Node 3 — Asset Search
# -----------------------------
asset_agent_app = None # Global variable to hold the compiled asset agent app

def node_asset_search(state: UI2CodeState) -> UI2CodeState:
    """
    For each asset requested by the planner, this node invokes the asset agent
    workflow to find and download the asset.
    """
    global asset_agent_app
    if not asset_agent_app:
        asset_agent_app = init_asset_agent_workflow() # Compile the asset agent graph once

    print("--- Starting Asset Search---")
    assets_to_find = state.plan_json.get("assets_needed", [])
    if not assets_to_find:
        print("No assets needed.")
        return state

    for asset_request in assets_to_find:
        component_id = asset_request['component_id']
        description = asset_request['description']
        bounding_box = asset_request['bounding_box']
        print(f"-> Finding asset for '{component_id}': {description}")

        # Get bounding box and convert to dimensions for the asset agent
        width = bounding_box.get('width', 512)
        height = bounding_box.get('height', 512)

        # Prepare the input for the asset agent workflow
        asset_agent_input = {
            "instructions": description,
            "bounding_box": (width, height),
            "user_asset_options": state.asset_paths
        }

        print(f"Invoking asset agent with input: {asset_agent_input}")

        # Invoke the asset agent
        result = asset_agent_app.invoke(asset_agent_input)

        final_path = result.get("final_asset_path")
        if final_path:
            state.asset_paths[component_id] = final_path
            msg = f"Asset found for {component_id}: {final_path}"
            state.messages.append(msg)
        else:
            msg = f"Asset search failed for {component_id}."
            state.messages.append(msg)
        print(f"{msg}")

    return state

# -----------------------------
# Node 4 — Codegen: HTML+CSS from plan
# -----------------------------
CODE_PROMPT = """You are a senior UI engineer.
Generate a SINGLE self-contained HTML file with <style> (no external CSS/JS).
Follow the plan exactly. Use semantic tags where sensible. Prefer CSS Grid/Flex.
Add minimal responsive rules (≤ 60 lines CSS if possible).

When generating an `<img>` tag for a component ID found in the `asset_paths` JSON, 
you MUST use the provided local file path in the `src` attribute. For all other images, use a placeholder.
"""

def node_codegen(state: UI2CodeState) -> UI2CodeState:
    assert state.plan_json and state.description_json, "Missing inputs for codegen"

    sys = {
        "role": "system",
        "content": [{"type": "text", "text": "You write production-quality HTML+CSS from structured UI plans."}],
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": CODE_PROMPT},
            {"type": "text", "text": "Component description JSON:"},
            {"type": "text", "text": json.dumps(state.description_json, ensure_ascii=False, indent=2)},
            {"type": "text", "text": "Ordered build plan JSON:"},
            {"type": "text", "text": json.dumps(state.plan_json, ensure_ascii=False, indent=2)},
            {"type": "text", "text": "Asset paths JSON (map of component_id to local file path):"},
            {"type": "text", "text": json.dumps(state.asset_paths, ensure_ascii=False, indent=2)},
            {"type": "text", "text": "Now return only the final HTML."},
        ],
    }


    # We can reuse Qwen2.5-VL for code generation; it’s text-only here.
    resp = qwen_vl.chat([sys, user], temperature=0.15, max_new_tokens=2500)
    # Light post-filter to keep just the HTML
    start = resp.find("<html")
    if start == -1:
        start = resp.find("<!DOCTYPE")
    end = resp.rfind("</html>")
    html = resp[start:end+7] if end != -1 and start != -1 else resp

    state.html_css = html
    state.messages.append("Codegen complete: produced HTML+CSS.")
    return state

# -----------------------------
# Conditional Router
# -----------------------------
def route_to_assets_or_codegen(state: UI2CodeState) -> str:
    """
    Checks the planner's output. If assets are needed, routes to the
    asset_search node. Otherwise, proceeds directly to codegen.
    """
    print("--- Routing after planner ---")
    if state.plan_json and state.plan_json.get("assets_needed"):
        print("Decision: Assets are needed. Routing to asset search.")
        return "asset_search"
    else:
        print("Decision: No assets needed. Routing to codegen.")
        return "codegen"
    
# -------------------------------------------------------------
# ALL ASSET AGENT CODE IS HERE
# -------------------------------------------------------------
def asset_search_node(state: AssetState) -> AssetState:
    """
    Searches for an asset using only the 'description' field from the state.
    This field is guaranteed to be populated by the preceding 'query_transformer_node'.
    """
    # This will now correctly use the refined query from the current run.
    description_to_search = state['instructions']
    image_url = search_for_asset(description_to_search)

    if image_url:
        # The unique filename generation is correct and remains unchanged.
        path = process_and_resize_image_from_url(image_url, state['bounding_box'], f"asset_{uuid4()}.png")
        return {"search_successful": True, "final_asset_path": path}
        
    return {"search_successful": False}

def user_asset_node(state: AssetState) -> AssetState:
    asset_path = state['user_asset_options'][0]
    print(f"Using user-provided asset: {asset_path}")
    return {"final_asset_path": asset_path}

def query_transformer_node(state: AssetState) -> dict:
    """Refines the user's instructions into a concise search query using the main VL model."""
    print("--- Query Transformer Node ---")

    sys = {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert at refining user requests into a concise 2-5 word search query for a photo API. You return ONLY the query, with no explanation or extra text."}]
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"REQUEST: {state['instructions']}\n\nREFINED QUERY:"}
        ]
    }

    # Use the global qwen_vl instance, the same one used by other nodes
    refined_query = qwen_vl.chat([sys, user], temperature=0.0, max_new_tokens=20)
    cleaned_query = refined_query.strip().strip("'\"")

    print(f"Refined query: '{cleaned_query}'")
    return {"description": cleaned_query}

def process_and_resize_image_from_url(image_url: str, bounding_box: Tuple[int, int], filename: str) -> Optional[str]:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Resize while maintaining aspect ratio to fit within the bbox
        img.thumbnail(bounding_box)
        output_path = os.path.join("Outputs/Assets", filename)
        img.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def search_for_asset(description: str) -> Optional[str]:
    print(f"Searching Pexels for: '{description}'")
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("Error: PEXELS_API_KEY not found in .env file.")
        return None
    headers = {"Authorization": api_key}
    params = {"query": description, "per_page": 1}
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data["photos"]:
            print(f"Found image: {data['photos'][0]['src']['original']}")
            return data["photos"][0]["src"]["original"]
        else:
            print("No images found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Pexels API: {e}")
        return None

def llm_router(state: AssetState) -> dict:
    # Simplified for this workflow since we aren't using user-provided assets
    if state.get("user_asset_options"):
         return {"route_decision": "use_user_asset"}
    return {"route_decision": "search"}

def post_search_router(state: AssetState) -> str:
    return END # Always end after a search attempt

def init_asset_agent_workflow():
    """Initializes and compiles the asset agent's StateGraph."""
    workflow = StateGraph(AssetState)

    # The entry point is now explicitly the query transformer.
    workflow.set_entry_point("query_transformer")

    workflow.add_node("query_transformer", query_transformer_node)
    workflow.add_node("search", asset_search_node)

    # The flow is now linear: always transform the query, then search.
    workflow.add_edge("query_transformer", "search")
    workflow.add_conditional_edges("search", post_search_router, {END: END})

    return workflow.compile()

# -----------------------------
# Build the LangGraph
# -----------------------------
workflow = StateGraph(UI2CodeState)
workflow.add_node("perception", node_perception)
workflow.add_node("planner", node_planner)

workflow.add_node("asset_search", node_asset_search)
workflow.add_node("codegen", node_codegen)

workflow.set_entry_point("perception")
workflow.add_edge("perception", "planner")
workflow.add_conditional_edges(
    "planner",
    route_to_assets_or_codegen,
    {
        "asset_search": "asset_search",
        "codegen": "codegen"
    }
)
workflow.add_edge("asset_search", "codegen")
workflow.add_edge("codegen", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# CLI Runner
# -----------------------------
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=False, help="Path to wireframe PNG/JPG", default="Images/asset_test.png")
    p.add_argument("--out_html", default="Outputs/generated_asset_test.html")
    args = p.parse_args()

    img_path = str(pathlib.Path(args.image).resolve())
    state = UI2CodeState(image_path=img_path)

    run_id = f"wireframe-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    final = app.invoke(state, config=config)

    # Save HTML
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(final['html_css'] or "<!-- empty -->")

    # Log
    print("\n--- PIPELINE MESSAGES ---")
    print("\n".join(final['messages']))
    print(f"\nSaved: {args.out_html}")
