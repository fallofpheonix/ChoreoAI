# Architecture & Roadmap Plan: ChoreoAI

## Phase 0 — Repo Bring-up (Week 0)
Tasks:
- Define build/test entrypoints
- Add dataset validation CLI
- Stabilize doc set and scope

Output:
```
Runnable scaffold + aligned docs
```

## Phase 1 — Motion Dataset Construction (Weeks 1–3)
Tasks:
- Collect/locate dance video dataset
- Implement pose extraction (MediaPipe/OpenPose)
- Convert to 3D skeletal format
- Build dataset loader + validators
- Add preprocessing (interpolation, smoothing, normalization)

Output:
```
Dynamic Motion Dataset ready for model ingestion
```

## Phase 2 — Multimodal Encoders (Weeks 3–5)
Tasks:
- Implement encoders:
```
text_encoder.py
image_encoder.py
audio_encoder.py
motion_encoder.py
```
- Train contrastive embedding space

Output:
```
Shared latent representation
```

## Phase 3 — Motion Generation Model (Weeks 5–7)
Tasks:
- Implement conditional generator (diffusion preferred)
- Train:
```
latent → motion sequence
```

Output:
```
Generative choreography model
```

## Phase 4 — Visualization and Artist Feedback (Week 7)
Tasks:
- Build visualization pipeline
- Render sequences
- Collect artist feedback

Tools:
```
Blender
Three.js
Matplotlib animation
```

Output:
```
AI-generated choreography sequences
```
