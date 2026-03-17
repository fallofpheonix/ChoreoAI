# ChoreoAI

ChoreoAI is a multimodal-to-motion pipeline for generating 3D dance pose sequences from text prompts.
It includes model inference, dataset lifecycle tooling, and a lightweight HTTP API.

## Project layout

```text
src/choreoai/
  api/        # HTTP transport layer (FastAPI app and schemas)
  config/     # environment-backed application settings
  core/       # domain entities and validation rules
  services/   # orchestration layer used by CLI/API
  encoders/   # representation learning models
  generator/  # diffusion model components
  utils/      # shared cross-cutting helpers
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run CLI:

```bash
choreoai --help
choreoai summarize --root data/dataset
```

Run API:

```bash
uvicorn choreoai.api:app --reload
```

Run tests:

```bash
pytest
```

## Key decisions

- Keep model code mostly unchanged and refactor around it via services.
- Preserve existing command behavior while reducing adapter-level duplication.
- Add practical validation and telemetry hooks without over-abstracting.

## Notes

- `.env.example` documents supported runtime environment variables.
- Dataset preprocessing still favors readability over maximum throughput.
