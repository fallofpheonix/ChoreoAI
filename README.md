# AI-Enabled Choreography - Dance Beyond Music

## Description
Recent years have seen the advent of AI-generated choreography using models trained on motion capture of a single dancer (see e.g. https://arxiv.org/abs/1907.05297), and last year with GSoC 2024, two contributors developed cutting-edge projects to understand improvisational dance duets through the lens of neural networks including GNNs and Transformers. However, many dance traditions view dance as far more than just a visual art, and understanding dance as only a movement prediction project risks overly reducing the perception of dance in digital form. Moreover, while existing multimodal dance embeddings focus primarily on pairings of music (e.g. “beats”) to movement, human movement can incorporate and be influenced by diverse modalities well beyond music including speech, imagery, writing, touch, architecture, proprioception, sculpture, and more. By exploring diverse multimodal embeddings of dance in an artist-driven framework, this project will imagine how to use AI to bring expansiveness, rather than reductiveness and conformity, into our digital renderings of dance.

## Duration
Total project length: 175 hours

## Expected Results
1. Create a dataset of dynamic point-cloud data corresponding to extracted motion capture poses from videos of dances
2. Craft a (semi-)supervised paradigm for constructing paired modalities corresponding to the dance movements, e.g. natural language descriptions, spoken word, music, visual art, 3D architectural renderings, and more
3. Design a multimodal embedding scheme for these highly diverse modalities
4. Train an any-to-any generative model that is able to translate an arbitrary modality input into 3D movement (and vice-versa)
5. Work closely with the dance artists to ensure that the model design is artist-led

## Requirements
Participants should be comfortable with standard data science software including Python, Git, Numpy, Matplotlib, and Pandas. Previous experience in Machine Learning, either in TensorFlow or PyTorch, is preferred. While previous experience in dance or the performing arts is not needed, an interest in the artistic and open-ended aesthetic dimensions of the project is required. Strong interpersonal & communication skills are essential.

## Project Difficulty Level
Hard

## Mentors
- Mariel Pettee (Lawrence Berkeley National Laboratory)
- Ilya Vidrin (Northeastern University)

Please DO NOT contact mentors directly by email. Instead, please email human.ai.choreo@gmail.com with subject line “Test Submission: AI Choreo” and include your CV and Github repository link. The mentors will then get in touch with you.

## Corresponding Project
ChoreoAI

## Participating Organizations
- LBNL
- Northeastern

## Current Repository Status
This repository currently implements the dataset bring-up stage:

- package scaffold via `pyproject.toml`
- dataset validation CLI
- dataset staging CLI for raw `.npy` pose arrays
- preprocessing CLI for interpolation, smoothing, and normalization
- sample raw pose artifact in `data/raw/sample_dance_pose.npy`

The multimodal encoder stack, generator, training loop, and visualization pipeline are still pending.

## Current Data Layout
Raw pose arrays:

```text
data/raw/
  sample_dance_pose.npy
```

Validated dataset layout:

```text
data/dataset/
  sequence_id/
    poses.npy
    text_prompt.txt         # optional
    image_reference.png     # optional
    audio.wav               # optional
```

Preprocessed dataset layout:

```text
data/processed/
  sequence_id/
    poses.npy
    text_prompt.txt         # optional
    image_reference.png     # optional
    audio.wav               # optional
```

## CLI Workflow
Bootstrap a dataset from raw pose arrays:

```bash
PYTHONPATH=src python3 -m choreoai.cli bootstrap-dataset --raw-root data/raw --root data/dataset
```

Validate dataset structure and pose tensor shape:

```bash
PYTHONPATH=src python3 -m choreoai.cli validate-dataset --root data/dataset
```

Summarize sequence-level dataset statistics:

```bash
PYTHONPATH=src python3 -m choreoai.cli summarize-dataset --root data/dataset
```

Preprocess dataset sequences for downstream modeling:

```bash
PYTHONPATH=src python3 -m choreoai.cli preprocess-dataset --root data/dataset --output-root data/processed
```

## Preprocessing Contract
Each sequence is processed in this order:

1. Linear interpolation along the time axis for non-finite values.
2. Odd-window moving average smoothing per joint coordinate.
3. Root-centering using the selected joint index.
4. Scale normalization by mean joint radius.

The output tensor remains `(T,K,3)` and is stored as `float32`.
