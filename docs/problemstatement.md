# Problem Statement: ChoreoAI

## Problem Statement
Current AI choreography systems treat dance as a **music-conditioned motion prediction task**, typically learning mappings from audio beats to skeletal motion sequences. This formulation reduces movement to a narrow signal-prediction problem and restricts the expressive and cultural dimensions of dance.

Dance is **multimodal**. Movement can be influenced by:

- Spoken language
- Visual imagery
- Architecture and spatial structure
- Poetry or written text
- Physical interaction and proprioception
- Music or environmental sound

Existing datasets and models fail to capture this multimodal structure. They typically rely on:

- Single dancer motion capture datasets
- Music-conditioned generation
- Fixed skeletal representations

This leads to models that generate **repetitive and structurally constrained motion**, often biased toward upright locomotion or simple dance gestures.

The core technical problem is:

> **How can we construct a multimodal representation of dance that enables translation between heterogeneous modalities and dynamic human movement without collapsing artistic intent into a single conditioning channel?**

The system must support **any-to-any modality translation**, including:

```
Text → Motion
Image → Motion
Speech → Motion
Architecture → Motion
Motion → Text
Motion → Image
```

This requires:

1. A dataset of **dynamic 3D skeletal point clouds extracted from dance videos**.
2. A **multimodal embedding space** aligning diverse artistic modalities.
3. A **generative model capable of decoding the latent representation into movement sequences.**
