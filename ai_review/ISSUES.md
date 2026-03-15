# AI Review: ChoreoAI

## Current Status
Documentation: ✅ Complete
Repository Structure: ✅ Cleaned (docs/ duplicate removed)
Core Model Code: ❌ Missing entirely
Scaffold: ✅ Build metadata + dataset bootstrap/summary/validation CLI added
Preprocessing: ✅ Interpolation, smoothing, and normalization pipeline added

---

## Missing
- Pose extraction script (MediaPipe or OpenPose) not implemented
- Encoders (text/image/audio/motion) not implemented
- Generator (diffusion or transformer) not implemented
- Training/eval pipeline not started
- Visualization script (skeleton animation) not created
- Model-ready dataset wrapper not implemented

## Mistakes / Problems
- Docs reference encoders and generator but no implementations exist
- Model choices (e.g., RoBERTa, ResNet/ViT) are not pinned in dependencies

## Next Actions
1. Implement `pose_extractor.py` using MediaPipe or OpenPose
2. Add a model-ready dataset wrapper over `data/processed`
3. Create skeleton animation visualization utility
4. Implement text and image encoders
5. Choose and download a larger dataset sample (e.g., AIST++)
