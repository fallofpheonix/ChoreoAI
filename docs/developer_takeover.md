# Developer Takeover Scan: ChoreoAI

## 1. Repository Scan
**Structure:**
- **Core Modules:** `src/choreoai/dataset.py`, `src/choreoai/preprocess.py`, `src/choreoai/cli.py`
- **Configuration:** `pyproject.toml`, `requirements.txt`
- **Data:** `data/raw/sample_dance_pose.npy`, `data/dataset/` (generated dataset layout), `data/processed/` (normalized output)
- **Docs:** `doc/problemstatement.md`, `proposal.md`, `constraints.md`, `roadmap.md`, `todo.md`

## 2. Documentation Review
Reviewed the GSoC proposal and roadmap. Project aims to be a generalized Multimodal AI Choreography Generator translating text, image, and audio into 3D skeletal motion sequences.

## 3. Project Objective
**Goal:** Generate responsive 3D dance motion from arbitrary input modalities.
**Output:** A trained conditional diffusion or transformer model evaluating multimodal embeddings and returning a temporal sequence of 3D skeletal points.

## 4. System Architecture
1. **Pose Extraction:** Transforms raw dance video into 3D skeleton sequences `M = {P_1, ..., P_T}`.
2. **Dataset Bootstrap:** Copies raw `.npy` pose tensors into sequence directories with optional paired modalities.
3. **Preprocessing:** Repairs non-finite values by time interpolation, applies moving-average smoothing, centers on a root joint, and normalizes body scale.
4. **Encoders:** Separate branches for Text (RoBERTa), Image (ViT), Audio (Spectrogram CNN), and Motion (Temporal Transformer).
5. **Alignment:** InfoNCE contrastive loss enforcing a shared latent space `z`.
6. **Generator:** Conditional diffusion decoding `p(M | z_input)`.

## 5. Existing Codebase Analysis
- Currently, the codebase consists of infrastructural skeleton code.
- `src/choreoai/dataset.py` defines a `DatasetIndex` parser to read sequence directories containing `poses.npy`, `text_prompt.txt`, `image_reference.png`, and `audio.wav`.
- `src/choreoai/dataset.py` also stages raw `.npy` arrays into dataset layout and summarizes sequence statistics.
- `src/choreoai/preprocess.py` performs interpolation, smoothing, and normalization over `(T,K,3)` pose tensors.
- `src/choreoai/cli.py` exposes dataset bootstrap, staging, preprocessing, validation, and summary commands.
- No ML models or training pipelines are implemented yet.

## 6. Dependency Configuration
- Needs modern deep learning packages (`torch`, `transformers`, `diffusers`). `requirements.txt` and `pyproject.toml` are primarily stubs right now (`numpy`).

## 7. Incomplete/Unstable Components
- `todo.md` lists the entire ML pipeline as remaining: pose extraction, visualization, encoders, generators, and train loops.

## 8. Tests and Evaluation
- No automated unit tests (`tests/` directory missing).
- Evaluation is currently limited to structural dataset checks, summary inspection, and preprocessing output sanity checks.

## 9. Assigned Task Clarification
Current immediate assignment is to extend the repo from preprocessing into actual model ingestion and baseline training code.

## 10. Continuous Documentation
Tracked via this `developer_takeover.md` file. The foundational structure is clean, but the project requires heavy modeling implementation.
