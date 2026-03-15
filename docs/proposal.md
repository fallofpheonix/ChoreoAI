# GSoC 2026 Proposal: ChoreoAI
## *Multimodal AI Choreography Generator: Text, Image, and Audio to 3D Motion*

---

## 1. Contact Information

| Field | Value |
|-------|-------|
| **Student Name** | Ujjwal Singh |
| **Email** | ujjosing@gmail.com |
| **GitHub** | github.com/fallofpheonix |
| **Time Zone** | IST (UTC+5:30) |

---

## 2. Synopsis

Most AI choreography systems treat dance as music-conditioned motion prediction, severely restricting expressive range. This project builds a **multimodal to motion translation system**: given any combination of text, image, speech, or music, generate 3D skeletal dance sequences. The approach extracts dynamic point clouds from dance videos, trains contrastive (InfoNCE) multimodal encoders to align modalities in a shared latent space, and decodes them into motion sequences via a conditional diffusion model.

---

## 3. Benefits to Community

- Provides choreographers and artists with a novel AI tool for creative ideation.
- Contributes a publicly available multimodal dance dataset and encoder suite.
- Advances the field of any-to-any cross-modal generation beyond unimodal audio → motion systems.

---

## 4. Technical Approach

### System Architecture

```
Dance Video → MediaPipe Pose Extraction → Dataset Bootstrap → Preprocessing → 3D Skeleton Dataset
Text/Image/Audio → Modality-specific Encoder → z ∈ R^d (shared space)
Motion Encoder → z ∈ R^d

Contrastive Alignment: InfoNCE(z_motion, z_text)

z → Diffusion Model → Motion sequence M = {P_1, ..., P_T}
where P_t ∈ R^(3K), K = number of joints
```

### Encoders

| Modality | Architecture |
|----------|-------------|
| Text | RoBERTa (frozen → finetuned) |
| Image | ViT-B/16 |
| Audio | Spectrogram CNN |
| Motion | Temporal Transformer |

### Generative Model

Conditional diffusion:
```
p(M | z_input) via DDPM with classifier-free guidance
```

---

## 5. Deliverables

| # | Deliverable | Required/Optional |
|---|------------|-------------------|
| 1 | Dance video pose extraction pipeline | Required |
| 2 | Multimodal encoder training (text + motion) | Required |
| 3 | Contrastive alignment on motion-text pairs | Required |
| 4 | Conditional diffusion / transformer generator | Required |
| 5 | Skeleton animation visualization | Required |
| 6 | Image and audio encoder integration | Optional |

---

## 6. Timeline (175 hours)

| Period | Activity |
|--------|----------|
| **Pre-bonding** | Study AIST++ dataset, motion diffusion literature |
| **Weeks 1–3** | Build pose extraction, dataset bootstrap, and preprocessing pipeline |
| **Weeks 4–5** | Train text and motion encoders with contrastive loss |
| **Weeks 6–8** | Train conditional motion generator (diffusion or transformer) |
| **Weeks 9–10** | Visualization + artist feedback loop |
| **Weeks 11–12** | Final evaluation, documentation, PR |

---

## 7. Related Work

- **AIST++** (Li et al., 2021): music-conditioned choreography dataset and baseline.
- **CLIP** (Radford et al., 2021): contrastive multimodal alignment paradigm.
- **MDM** (Tevet et al., 2022): motion diffusion model baseline.

---

## 8. About Me

**Ujjwal Singh** | ujjosing@gmail.com | [GitHub](https://github.com) | [LinkedIn](https://linkedin.com) | VIT University, B.Tech CS (2023–2027) | IST (UTC+5:30)

- **Computer Vision & DL:** TerraHerb plant disease classifier using CNN transfer learning — demonstrates experience with feature extraction architectures directly applicable to the multimodal encoders in this project.
- **Mobile Spatial Systems:** Built *UDIE*, an iOS disruption intelligence app with spatial heatmap visualization using MapKit — shows experience with spatial data representation, relevant to 3D motion visualization in ChoreoAI.
- **Technical Skills:** Python, TensorFlow, Keras, NumPy, Pandas, Swift, SwiftUI, Flutter, Git.
- **Interested in:** Generative models, diffusion models, and multimodal learning — areas I have been actively studying to prepare for this project.
