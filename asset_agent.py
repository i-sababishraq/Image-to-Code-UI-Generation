import os
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import RetryOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# state and route definitions
class AssetState(TypedDict):
    instructions: str
    description: str
    bounding_box: Tuple[int, int]
    user_asset_options: List[str]
    search_successful: bool
    final_asset_path: Optional[str]
    route_decision: str
    error: Optional[str]

class RouteQuery(BaseModel):
    destination: str = Field(description="The next node to route to. Must be one of: 'search' or 'use_user_asset'.")

# Graph nodes
def search_node(state: AssetState) -> AssetState:
    image_url = search_for_asset(state['description'])
    if image_url:
        path = process_and_resize_image_from_url(image_url, state['bounding_box'], "searched_asset.png")
        return {"search_successful": True, "final_asset_path": path}
    return {"search_successful": False}

def user_asset_node(state: AssetState) -> AssetState:
    asset_path = state['user_asset_options'][0]
    print(f"✅ Using user-provided asset: {asset_path}")
    return {"final_asset_path": asset_path}

def query_transformer_node(state: AssetState) -> dict:
    """Refines the user's instructions into a concise search query."""
    print("--- QUERY TRANSFORMER NODE ---")
    
    llm = get_llm_pipeline()
    
    prompt = ChatPromptTemplate.from_template(
        "Your sole task is to extract the essential keywords from a user's request to create a concise search query for a photo API.\n"
        "Do not add any explanation, conversational text, or examples. Your response must be ONLY the refined query itself.\n\n"
        "## USER REQUEST:\n"
        "{instructions}\n\n"
        "## REFINED QUERY:"
    )
    
    chain = prompt | llm
    
    refined_query = chain.invoke({"instructions": state["instructions"]})
    cleaned_query = refined_query.strip().strip("'\"")
    print(f"Refined query: '{cleaned_query}'")
    
    return {"description": cleaned_query}

# Helper functions
def convert_bbox_to_dimensions(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    xmin, ymin, xmax, ymax = bbox
    return (xmax - xmin, ymax - ymin)

def process_and_resize_image_from_url(image_url: str, bounding_box: Tuple[int, int], filename: str) -> Optional[str]:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.thumbnail(bounding_box)
        output_path = os.path.join("assets", filename)
        img.save(output_path)
        print(f"🖼️ Image saved and resized to {output_path}")
        return output_path
    except Exception as e: return None

def search_for_asset(description: str) -> Optional[str]:
    """
    Performs a real image search using the Pexels API.
    """
    print(f"🔎 Searching Pexels for an asset described as: '{description}'")
    
    # 1. Get your API key from the environment variables
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("Error: PEXELS_API_KEY not found. Please add it to your .env file.")
        return None

    # 2. Set up the API request
    headers = {
        "Authorization": api_key
    }
    params = {
        "query": description,
        "per_page": 1, # We only need one image
        "orientation": "landscape"
    }
    url = "https://api.pexels.com/v1/search"

    try:
        # 3. Make the API call
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)

        # 4. Parse the response and extract the image URL
        data = response.json()
        if data["photos"]:
            # We are getting the URL for the original, high-quality image
            image_url = data["photos"][0]["src"]["original"]
            print(f"-> Search successful! Found image URL: {image_url}")
            return image_url
        else:
            print("-> Search failed. Pexels did not return any images for this query.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling Pexels API: {e}")
        return None

def get_llm_pipeline():
    """Helper function to load and cache the LLM pipeline."""
    global llm_pipeline_cache
    if not llm_pipeline_cache:
        print("Loading local model for the first time... (this may take a moment)")
        model_id = "Qwen/Qwen3-8B-Base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, return_full_text=False)
        llm_pipeline_cache['llm'] = HuggingFacePipeline(pipeline=pipe)
    return llm_pipeline_cache['llm']

        
# Global cache for the loaded model and pipeline to avoid reloading on every call
llm_pipeline_cache = {}

def llm_router(state: AssetState) -> dict:
    print("--- Transformers LLM ROUTER ---")
    llm = get_llm_pipeline()

    if state.get("user_asset_options"):
        system_context = "Context: The user has provided their own assets. If they mention using their own files, prioritize the 'use_user_asset' tool."
    else:
        system_context = "Context: The user has NOT provided any assets. The 'use_user_asset' tool is unavailable."
    print(system_context)

    parser = JsonOutputParser(pydantic_object=RouteQuery)
    retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm, max_retries=3)
    prompt_template = """You are an expert at routing a user's request to the correct tool. Your output MUST be a valid JSON object.

    {system_context}

    The available tools are:
    - `search`: To find an asset online based on a description. This is the default.
    - `use_user_asset`: ONLY if the user explicitly mentions using their own file or a pre-existing asset.

    User instructions: "{instructions}"
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm
    prompt_value = prompt.invoke({
        "instructions": state['instructions'], 
        "system_context": system_context
    })
    completion = chain.invoke({
        "instructions": state['instructions'],
        "system_context": system_context
    })
    parsed_result = retry_parser.parse_with_prompt(
        completion=completion, 
        prompt_value=prompt_value
    )
    destination = parsed_result['destination']
    
    print(f"LLM Decision: Route to '{destination}'")
    return {"route_decision": destination}

def post_search_router(state: AssetState) -> str:
    """Routes to generation on failure, otherwise ends."""
    print("--- Post-Search Router ---")
    if state['search_successful']:
        print("Decision: Search was successful, end.")
        return END
    else:
        print("Decision: Search failed, end.")
        return END

# Build graph
workflow = StateGraph(AssetState)
workflow.add_node("query_transformer", query_transformer_node)
workflow.add_node("llm_router", llm_router)
workflow.add_node("search", search_node)
workflow.add_node("user_asset_node", user_asset_node)
workflow.set_entry_point("llm_router")
workflow.add_conditional_edges(
    "llm_router",
    lambda state: state["route_decision"],
    {
        "search": "query_transformer",  # On 'search', go to the transformer first
        "use_user_asset": "user_asset_node" # On 'use_user_asset', go directly to that node
    }
)
workflow.add_edge("query_transformer", "search")
workflow.add_conditional_edges("search", post_search_router, {END: END})
workflow.add_edge("user_asset_node", END)
app = workflow.compile()

# Run Examples
if __name__ == "__main__":
    if not os.path.exists("assets"): os.makedirs("assets")

    # Example 1: General request routed to 'search'
    print("\n\n--- RUNNING EXAMPLE 1: Transformers routes to SEARCH ---")
    inputs_search = {
        "instructions": "I need a cool picture of a robot.",
        "description": "a cool picture of a robot",
        "bounding_box": (512, 512), "user_asset_options": []
    }
    result = app.invoke(inputs_search)
    print("Final State (Run 1):", result)

    # Example 2: Explicit request routed to 'use_user_asset'
    print("\n\n--- RUNNING EXAMPLE 2: Transformers routes to USE_USER_ASSET ---")
    if not os.path.exists("assets/test.jpg"):
        Image.new('RGB', (100, 100), color = 'blue').save('assets/test.jpg')
        
    inputs_user = {
        "instructions": "Please use the banner I provided for this component.",
        "description": "company banner",
        "bounding_box": (512, 512), "user_asset_options": ["assets/test.jpg"]
    }
    result = app.invoke(inputs_user)
    print("Final State (Run 2):", result)