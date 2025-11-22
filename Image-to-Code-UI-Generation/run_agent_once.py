"""
Entry point invoked by the React app. It reads a JSON request from stdin,
runs the Azure GPT + Pexels-powered pipeline, then prints a JSON response.

Example request payload written to stdin:
{
  "prompt": "Generate code for this layout",
  "image_path": "C:/path/to/wireframe.png",
  "find_assets": true
}
"""

from __future__ import annotations

import argparse
import io
import base64
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from contextlib import redirect_stdout, redirect_stderr


DEFAULT_PROMPT = "Generate production-ready HTML/CSS for this wireframe."
DEFAULT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", os.getenv("DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")))
DEFAULT_FIND_ASSETS = os.getenv("AGENT_FIND_ASSETS", "true").lower() in {"1", "true", "yes"}
DEFAULT_REFINE_ITERS = int(os.getenv("AGENT_REFINE_MAX_ITERS", "3"))
DEFAULT_REFINE_THRESHOLD = int(os.getenv("AGENT_REFINE_THRESHOLD", "8"))
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(os.getenv("AGENT_OUTPUT_ROOT", "Outputs/AgentRuns")).resolve()


@dataclass
class AgentRequest:
    prompt: str
    image_path: Path
    find_assets: bool = DEFAULT_FIND_ASSETS
    vision_deployment: str = os.getenv("VISION_DEPLOYMENT", DEFAULT_DEPLOYMENT)
    text_deployment: str = os.getenv("TEXT_DEPLOYMENT", DEFAULT_DEPLOYMENT)
    refine_max_iters: int = DEFAULT_REFINE_ITERS
    refine_threshold: int = DEFAULT_REFINE_THRESHOLD
    run_id: str = uuid.uuid4().hex


def _read_request_from_stdin() -> Optional[AgentRequest]:
    if sys.stdin.closed or sys.stdin is None or sys.stdin.isatty():
        return None
    raw = sys.stdin.read().strip()
    if not raw:
        return None
    payload = json.loads(raw)
    return _request_from_dict(payload)


def _request_from_dict(payload: Dict[str, Any]) -> AgentRequest:
    prompt = payload.get("prompt") or DEFAULT_PROMPT
    image_path = payload.get("image_path")
    if not image_path:
        raise ValueError("The request must include an 'image_path'.")
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    return AgentRequest(
        prompt=prompt,
        image_path=path,
        find_assets=bool(payload.get("find_assets", DEFAULT_FIND_ASSETS)),
        vision_deployment=payload.get("vision_deployment") or os.getenv("VISION_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        text_deployment=payload.get("text_deployment") or os.getenv("TEXT_DEPLOYMENT", DEFAULT_DEPLOYMENT),
        refine_max_iters=int(payload.get("refine_max_iters", DEFAULT_REFINE_ITERS)),
        refine_threshold=int(payload.get("refine_threshold", DEFAULT_REFINE_THRESHOLD)),
        run_id=payload.get("run_id") or uuid.uuid4().hex,
    )


def _request_from_cli(argv: list[str]) -> Optional[AgentRequest]:
    if len(argv) <= 1:
        return None

    parser = argparse.ArgumentParser(description="Manually invoke the GPT/Pexels pipeline.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image", required=True, help="Path to the input wireframe image.")
    parser.add_argument("--find-assets", action="store_true", default=DEFAULT_FIND_ASSETS)
    parser.add_argument("--vision-deployment", default=os.getenv("VISION_DEPLOYMENT", DEFAULT_DEPLOYMENT))
    parser.add_argument("--text-deployment", default=os.getenv("TEXT_DEPLOYMENT", DEFAULT_DEPLOYMENT))
    parser.add_argument("--refine-max-iters", type=int, default=DEFAULT_REFINE_ITERS)
    parser.add_argument("--refine-threshold", type=int, default=DEFAULT_REFINE_THRESHOLD)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv[1:])
    return AgentRequest(
        prompt=args.prompt,
        image_path=Path(args.image).expanduser().resolve(),
        find_assets=args.find_assets,
        vision_deployment=args.vision_deployment,
        text_deployment=args.text_deployment,
        refine_max_iters=args.refine_max_iters,
        refine_threshold=args.refine_threshold,
        run_id=args.run_id or uuid.uuid4().hex,
    )


def _ensure_output_paths(req: AgentRequest) -> Dict[str, Path]:
    run_folder = OUTPUT_ROOT / f"run-{req.run_id}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return {
        "base": run_folder,
        "html": run_folder / "generated.html",
        "brief": run_folder / "brief.txt",
        "reldesc": run_folder / "relative_description.txt",
    }


def _read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _load_preview_data(html_path: Path) -> Optional[Dict[str, Any]]:
    directory = html_path.parent
    if not directory.exists():
        return None

    # Prefer iteration screenshots (generated during scoring), otherwise pick any PNG.
    candidates = sorted(directory.glob(f"{html_path.stem}_iter*.png"), reverse=True)
    if not candidates:
        candidates = sorted(directory.glob("*.png"), reverse=True)
    if not candidates:
        return None

    image_path = candidates[0]
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "name": image_path.name,
        "size": image_path.stat().st_size,
        "dataUrl": f"data:image/png;base64,{encoded}",
    }


def _run_pipeline(req: AgentRequest) -> Dict[str, Any]:
    paths = _ensure_output_paths(req)
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    with redirect_stdout(captured_out), redirect_stderr(captured_err):
        from agent_azure_vlm_tools import helper_run_azure_vlm_pipeline

        payload = helper_run_azure_vlm_pipeline(
            image_path=str(req.image_path),
            find_assets=req.find_assets,
            out_html_path=str(paths["html"]),
            out_brief_path=str(paths["brief"]),
            out_reldesc_path=str(paths["reldesc"]),
            vision_deployment=req.vision_deployment,
            text_deployment=req.text_deployment,
            refine_max_iters=req.refine_max_iters,
            refine_threshold=req.refine_threshold,
        )
    result = json.loads(payload)
    if result.get("status") != "success":
        message = result.get("message", "Azure VLM pipeline returned an error.")
        raise RuntimeError(message)

    html_path = Path(result.get("final_html_path", paths["html"]))
    code = _read_text_if_exists(html_path)
    plan = _read_text_if_exists(paths["reldesc"])
    reasoning = _read_text_if_exists(paths["brief"])
    try:
        repo_relative = html_path.resolve().relative_to(REPO_ROOT)
        web_path = "/" + repo_relative.as_posix()
    except ValueError:
        web_path = None

    reasoning_lines = [line for line in (reasoning.splitlines() if reasoning else []) if line.strip()]
    preview = _load_preview_data(html_path)

    logs = result.get("messages", [])
    captured_logs = []
    if captured_out.getvalue().strip():
        captured_logs.append(captured_out.getvalue().strip())
    if captured_err.getvalue().strip():
        captured_logs.append(captured_err.getvalue().strip())

    logs.extend(captured_logs)
    logs.append(f"HTML saved to: {html_path}")
    logs.append(f"Relative description saved to: {paths['reldesc']}")
    logs.append(f"Brief saved to: {paths['brief']}")

    response_body: Dict[str, Any] = {
        "plan": plan,
        "reasoning_steps": reasoning_lines,
        "code": code,
        "feedback": logs[-2] if len(logs) >= 2 else "Pipeline completed.",
        "error": None,
        "logs": "\n".join(logs),
        "final_html_path": str(html_path),
        "final_html_web_path": web_path,
    }

    if preview:
        response_body["preview_image"] = preview

    return response_body


def _emit_response(success: bool, payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({"success": success, **payload}, ensure_ascii=False))
    sys.stdout.flush()


def main(argv: list[str]) -> None:
    sys.stderr.write(f"DEBUG: Available Env Vars: {list(os.environ.keys())}\n")
    try:
        request = _request_from_cli(argv) or _read_request_from_stdin()
        if not request:
            raise ValueError(
                "No input received. Pass CLI arguments (--image ...) or write a JSON payload to stdin."
            )
        result = _run_pipeline(request)
        _emit_response(True, {"result": result})
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "logs": traceback.format_exc(),
        }
        _emit_response(False, error_payload)


if __name__ == "__main__":
    main(sys.argv)
