# GSoC 2026 Proposal: ChoreoAI
## Multimodal Dance Representation and Cross-Modal Motion Generation

---

### 1. Project Title
**ChoreoAI: A Framework for Multimodal Dance Representation and Generative Choreography**

---

### 2. Abstract / Overview
Traditional AI choreography systems often reduce dance to a music-conditioned movement prediction task. While effective for rhythm-based genres, this approach ignores the vast spectrum of choreographic inspirations, including language, architecture, and visual art. **ChoreoAI** aims to bridge this gap by building a robust, artist-led framework for multimodal dance representation. 

The project focuses on three core pillars:
1.  **Dataset Engineering**: Developing a principled pipeline to extract 3D point-cloud motion from mono-video and aligning it with diverse modalities (text, audio, images).
2.  **Multimodal Alignment**: Implementing a contrastive learning scheme to project movement and auxiliary modalities into a shared latent space.
3.  **Cross-Modal Synthesis**: Training a conditional generative model capable of synthesizing human motion from arbitrary modality inputs.

By the end of the 10-12 week period, ChoreoAI will provide the **HumanAI** organization with a research-ready infrastructure that empowers dance artists to use AI as an expansive creative partner rather than a reductive tool.

---

### 3. Problem Statement
The current state of "AI in Dance" suffers from several critical limitations:
*   **Music-Centric Bias**: Most models assume dance is purely a response to music beats, neglecting traditions where dance is driven by poetry, spatial geometry, or internal narrative.
*   **Lack of Diverse Alignment**: There is no standard framework for aligning 3D skeletal data with non-audio modalities (e.g., architectural renderings or natural language).
*   **Dataset Fragmentation**: Motion data is often siloed, lacks validated metadata, and is difficult for artists to use without deep technical expertise.

**Technical Challenges:**
*   **Temporal Synchronization**: Aligning sporadic textual prompts or static images with time-series motion data.
*   **Representation Collapse**: Ensuring the latent space preserves the nuance of movement without being dominated by a single modality (e.g., text).
*   **Artist-in-the-Loop**: Designing interfaces that allow choreographers to intuitively "prompt" movement using their own artistic vocabulary.

This project is vital for HumanAI because it moves beyond "black-box" sequence prediction toward an interpretable, multimodal dialogue between human intention and machine generation.

---

### 4. Project Constraints and Assumptions
*   **Compute Limitations**: Training will be optimized for a single high-end consumer GPU (e.g., RTX 3090/4090) within the GSoC timeframe.
*   **Data Sourcing**: We assume access to a curated set of dance videos; motion capture will be emulated via high-fidelity pose extraction (MediaPipe/ViTPose).
*   **Modality Scoping**: While "any-to-any" is the long-term goal, the GSoC baseline will prioritize **Text <-> Motion** and **Image <-> Motion** alignment.
*   **Skeletal Representation**: We will use a simplified 24-joint H-Anim or SMPL-compatible skeletal model to ensure compatibility with standard 3D engines (Blender/Unity).

---

### 5. Proposed Solution
#### 5.1 System Architecture
ChoreoAI adopts a three-stage architecture:
1.  **Extraction Layer**: Converts raw video into $(T, J, 3)$ point-cloud tensors using a robust pose-estimation backend.
2.  **Contrastive Alignment Layer (CAL)**: Uses Dual-Stream Transformers to project motion and auxiliary modalities (Text/Image) into a shared hypersphere.
3.  **Generative Decoding Layer (GDL)**: A conditional Motion Diffusion Model (MDM) that denoises gaussian noise into coherent skeletal sequences guided by the shared latent embedding.

#### 5.2 Data Flow
1.  **Input**: Video + Paired Modality (e.g., a text prompt like "serpentine and fluid").
2.  **Processing**: Pose extractor generates skeletal joints; Text encoder generates a semantic vector.
3.  **Closeness**: The Contrastive Loss forces the serpentine motion vector to be "close" to the serpentine text vector in latent space.
4.  **Generation**: A user provides a new prompt -> Encoder finds the latent vector -> Diffusion model generates a matching 3D sequence.

---

### 6. Technical Methodology
#### 6.1 Core Modules
*   `choreoai.extract`: Video preprocessing, pose extraction, and Kalman filtering for temporal smoothing.
*   `choreoai.align`: Implementation of the **Contrastive Dance-Modality Pre-training (CDMP)** loss.
*   `choreoai.generate`: A Diffusion Transformer (DiT) backbone for 3D motion synthesis.
*   `choreoai.viz`: Real-time OpenGL or Matplotlib-based skeletal visualizers.

#### 6.2 Implementation Strategy
*   **Motion Encoder**: ST-GCN (Spatio-Temporal Graph Convolutional Network) to capture joint hierarchies.
*   **Text/Image Encoders**: Freezing pre-trained backbones (CLIP/ViT) and training lightweight projection heads to save compute.
*   **Denoising Algorithm**: Using a standard DDPM (Denoising Diffusion Probabilistic Models) scheduler for motion generation.

#### 6.3 Evaluation Metrics
*   **FID (Frechet Inception Distance)**: Quantifying the quality/realism of generated motion.
*   **R-Precision**: Measuring how well the model can "retrieve" the correct text prompt for a given motion sequence.
*   **Foot-Sliding Score**: A physics-based heuristic to ensure grounded, realistic movement.

---

### 7. Implementation Plan
*   **Phase 1 (Weeks 1-3): Data Foundation**: Build the extraction pipeline and define the multimodal schema. Get artist feedback on "prompt" vocabulary.
*   **Phase 2 (Weeks 4-6): Embedding Alignment**: Implement the contrastive encoders. Deliver a working retrieval system (Motion-to-Text).
*   **Phase 3 (Weeks 7-9): Generative Modeling**: Train the conditional diffusion model. Focus on stability and temporal coherence.
*   **Phase 4 (Weeks 10-12): Integration & Documentation**: Finalize the API, build the README, and conduct a "virtual performance" demo for the HumanAI mentors.

---

### 8. Project Roadmap (GSoC Timeline)
*   **Community Bonding (May 4 - June 1)**: Sync with LBNL/Northeastern mentors. Refine Joint-List and Skeletal Scale.
*   **Weeks 1-2**: Extraction API. MediaPipe integration + Data cleaning.
*   **Week 3**: **Milestone 1**: 3D Point-Cloud dataset of 10+ hours of dance.
*   **Weeks 4-5**: Model training for Contrastive Alignment.
*   **Week 6**: **Midterm Evaluation**: Validation of retrieval metrics.
*   **Weeks 7-9**: Implementing the Motion Diffusion Model. Training on text-conditioning.
*   **Week 10**: Fine-tuning on secondary modalities (Image/Audio).
*   **Week 11**: Qualitative reviews with dance artists; metrics calculation.
*   **Week 12**: **Final Evaluation**: Final report, code cleanup, and GSoC submission.

---

### 9. Repository README Draft
```markdown
# ChoreoAI 🩰✨
> Multimodal Dance Representation & Generative Choreography

ChoreoAI is a toolkit developed for **HumanAI** to enable "any-to-any" translation between dance movement and artistic modalities (Text, Image, Architecture).

## Key Features
- 🎥 **Pose Extraction**: Mono-video to 3D skeletal data.
- 🧠 **Shared Latent Space**: Contrastive alignment of motion with language and visuals.
- 🧠 **Generative Diffusion**: Prompt-based 3D dance synthesis.
- 🎨 **Artist-Friendly**: Designed for choreographers, by technologists.

## Quick Start
```bash
pip install choreoai
choreoai preprocess --video input.mp4 --prompt "serpentine"
python -m choreoai.train.diffusion --config default.yaml
```

## Architecture
- **Pose Backend**: MediaPipe / ViTPose
- **Alignment**: Dual-Transformer (InfoNCE Loss)
- **Generation**: Diffusion Transformer (DiT)
```

---

### 10. Expected Outcomes and Impact
*   **A Standardized Dataset**: The first open-source dataset linking 3D dance to non-musical artistic prompts.
*   **State-of-the-Art Benchmarks**: Baseline metrics for multimodal choreography retrieval.
*   **HumanAI Benefit**: A tool that shifts the narrative from "AI replacing dancers" to "AI enhancing choreographic expansion."

---

### 11. Risks and Mitigation Strategies
*   **Data Scarcity**: *Risk*: Limited paired data for specific modalities. *Mitigation*: Data augmentation via synthetic prompt generation using LLMs (GPT-4) to describe motion.
*   **Temporal Jitter**: *Risk*: Extracted poses might "flicker." *Mitigation*: Implementation of Savitzky-Golay filtering and temporal loss constraints during training.

---

### 12. Future Work
*   **Interactive VR/AR**: Real-time generation of choreographic ghosts in immersive environments.
*   **SMPL-X Integration**: Moving from point-clouds to full-body mesh avatars with expressive fingers and faces.
*   **Robotic Choreography**: Exporting ChoreoAI latents to control humanoid robots on stage.

---
**Prepared for**: Mariel Pettee (LBNL) & Ilya Vidrin (Northeastern)
**Organization**: HumanAI
