# Image-to-Code-UI-Generation
Multimodal AI Agent for Image-to-Code UI Generation - CAP 6411 Computer Vision Systems Group Project

## Development Timeline & File Order

The project progressed through a set of sprints. The following ordered list reflects the historical development and authorship you provided, with `agent_azure_vlm_tools.py` as the most recent integration (final iteration).

- **Phase 1 — Initial data loaders & fix scripts (initial sprint):**
	- `dataloader.py` — core Web2Code dataloader
	- `dataloader_hf.py` — HuggingFace-style dataloader utilities
	- `fix_eval_image_paths.py` — evaluation image path fixup script

- **Phase 2 — Core agent and supporting modules (developed by Tina):**
	- `agent.py` — primary agent entry and orchestration
	- `nodes.py` — processing nodes used by the agent
	- `utils.py` — shared helper utilities
	- `constants.py` — project constants and config keys
	- `run_agent_once.py` — convenience runner for ad-hoc runs

- **Phase 3 — Azure VLM parallel agent (developed by Ashmal):**
	- `agent_azure_vlm.py` — Azure VLM specific agent or parallel implementation

- **Phase 4 — Asset agent (developed by Tina):**
	- `asset_agent.py` — asset-focused agent functionality

- **Phase 5 — Integration & final iteration (integration by Tina):**
	- `agent_azure_vlm_tools.py` — integration layer that combines components and represents the most recent, final iteration

- **Phase 6 — Frontend integration (integration by Megan):**
	- `run_agent_once.py` — integration layer that wraps the final integration into the format to connect with the frontend 

- **Other files / repo assets:**
	- `.env` — environment vars (not tracked in source control by default)
	- `requirements.txt` — Python dependencies
	- `uploads/` — directory for uploaded assets
	- `README.md` — this file

## Test Web2Code wireframes
Note: This was set up for finetuning, which may be a future direction to improve results.

### Setup
Download Web2Code dataset manually.
```
conda create -n Image2Code python=3.10
pip install -r requirements.txt
```
### Training
See dataloader.py, Web2CodeDataLoader. This function will iterate through the json data and zipped images.

### Evaluation
Run fix_eval_image_paths.py to generate Web2Code_eval_fixed.jsonl.
See dataloader.py, Web2CodeEvalDataLoader. This function will iterate through the jsonl data and zipped images.
