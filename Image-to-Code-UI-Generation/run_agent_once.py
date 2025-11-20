#!/usr/bin/env python3
"""
Interface script for Image2Code frontend.
Reads JSON from stdin: {"prompt": "...", "image_path": "..."}
Outputs JSON to stdout matching frontend expectations.
"""

import sys
import json
import os
import tempfile
import pathlib
from typing import Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the agent
from agent_azure_vlm import (
    CodeRefineState,
    app,
    get_client
)

def main():
    """Main entry point - reads from stdin, processes, writes to stdout."""
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)
        prompt = input_data.get("prompt", "").strip()
        image_path = input_data.get("image_path")
        
        if not image_path:
            output_error("No image_path provided in input")
            return
        
        if not os.path.exists(image_path):
            output_error(f"Image file not found: {image_path}")
            return
        
        # Check Azure OpenAI configuration
        endpoint = os.getenv("ENDPOINT_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if not endpoint or not api_key:
            output_error(
                "Azure OpenAI not configured. Set ENDPOINT_URL (or AZURE_OPENAI_ENDPOINT) "
                "and AZURE_OPENAI_API_KEY environment variables."
            )
            return
        
        # Get deployment names from environment or use defaults
        vision_deployment = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4.1-mini")
        text_deployment = os.getenv("AZURE_OPENAI_TEXT_DEPLOYMENT", "gpt-4.1-mini")
        
        # Create temporary output directory
        output_dir = pathlib.Path(tempfile.mkdtemp(prefix="image2code_"))
        out_rel_desc = str(output_dir / "relative_desc.txt")
        out_brief = str(output_dir / "ui_brief.txt")
        out_html = str(output_dir / "generated_page.html")
        
        # Create state for the agent
        state = CodeRefineState(
            image_path=image_path,
            out_rel_desc=out_rel_desc,
            out_brief=out_brief,
            out_html=out_html,
            vision_deployment=vision_deployment,
            text_deployment=text_deployment,
            reldesc_tokens=int(os.getenv("RELDESC_TOKENS", "700")),
            brief_tokens=int(os.getenv("BRIEF_TOKENS", "1100")),
            code_tokens=int(os.getenv("CODE_TOKENS", "2200")),
            judge_tokens=int(os.getenv("JUDGE_TOKENS", "900")),
            temp=float(os.getenv("TEMPERATURE", "0.12")),
            refine_max_iters=int(os.getenv("REFINE_MAX_ITERS", "3")),
            refine_threshold=int(os.getenv("REFINE_THRESHOLD", "8")),
            shot_width=int(os.getenv("SHOT_WIDTH", "1536")),
            shot_height=int(os.getenv("SHOT_HEIGHT", "900"))
        )
        
        # Run the agent workflow
        import uuid
        run_id = f"wireframe-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": run_id}}
        
        result = app.invoke(state, config=config)
        
        # Extract results - handle both dict and dataclass return types
        # LangGraph may return dict or CodeRefineState instance
        if isinstance(result, dict):
            plan = result.get("rel_desc", "") or ""
            reasoning_steps = result.get("messages", []) or []
            code = result.get("html", "") or ""
            scores = result.get("scores")
        else:
            # CodeRefineState dataclass instance
            plan = getattr(result, "rel_desc", "") or ""
            reasoning_steps = getattr(result, "messages", []) or []
            code = getattr(result, "html", "") or ""
            scores = getattr(result, "scores", None)
        
        feedback = ""
        error = None
        
        # Extract feedback from scores if available
        if scores and isinstance(scores, dict):
            feedback = scores.get("feedback", "")
            if not feedback and "raw" in scores:
                # Try to extract feedback from raw report
                feedback = scores.get("raw", "")
        
        # Combine all messages as logs
        logs = "\n".join(reasoning_steps) if reasoning_steps else ""
        
        # Format response for frontend
        output_success({
            "plan": plan,
            "reasoning_steps": reasoning_steps if isinstance(reasoning_steps, list) else [reasoning_steps] if reasoning_steps else [],
            "code": code,
            "feedback": feedback,
            "error": error,
            "logs": logs
        }, logs)
        
    except json.JSONDecodeError as e:
        output_error(f"Invalid JSON input: {str(e)}")
    except ImportError as e:
        import traceback
        error_msg = f"Python import error - missing dependency: {str(e)}"
        error_logs = traceback.format_exc()
        output_error(error_msg, error_logs)
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_logs = traceback.format_exc()
        output_error(error_msg, error_logs)

def output_success(result: Dict[str, Any], logs: str = ""):
    """Output success response in expected format."""
    response = {
        "success": True,
        "result": result,
        "logs": logs
    }
    print(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()

def output_error(error_message: str, logs: str = ""):
    """Output error response in expected format."""
    response = {
        "success": False,
        "error": error_message,
        "logs": logs
    }
    print(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()
    sys.exit(1)

if __name__ == "__main__":
    main()

