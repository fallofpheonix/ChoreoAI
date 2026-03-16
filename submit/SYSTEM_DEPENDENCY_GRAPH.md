# System Dependency Graph

```mermaid
graph TD
    api[api.py] --> inference[inference.py]
    cli[cli.py] --> inference
    cli --> pose_extractor[pose_extractor.py]
    cli --> dataset[dataset.py]
    cli --> dataset_index[dataset_index.py]
    cli --> preprocess_np[preprocess_np.py]
    
    inference --> text_encoder[encoders/text_encoder.py]
    inference --> denoiser[generator/denoising_transformer.py]
    inference --> scheduler[generator/diffusion_scheduler.py]
    
    train_gen[train_generator.py] --> torch_dataset[torch_dataset.py]
    train_gen --> denoiser
    train_gen --> scheduler
    train_gen --> text_encoder
    train_gen --> losses[losses.py]
    
    train_con[train_contrastive.py] --> torch_dataset
    train_con --> encoders[encoders/*]
    train_con --> losses
    
    torch_dataset --> dataset_index
    
    pose_extractor --> mediapipe[MediaPipe]
    pose_extractor --> opencv[OpenCV]
    
    preprocess_np --> dataset_index
```

## Internal Dependencies
- `choreoai.api` -> `choreoai.inference`
- `choreoai.inference` -> `choreoai.encoders.text_encoder`, `choreoai.generator.denoising_transformer`, `choreoai.generator.diffusion_scheduler`
- `choreoai.train_generator` -> `choreoai.torch_dataset`, `choreoai.generator.*`, `choreoai.encoders.text_encoder`, `choreoai.losses`
- `choreoai.train_contrastive` -> `choreoai.torch_dataset`, `choreoai.encoders.*`, `choreoai.losses`

## External Dependencies
- `torch`, `torchvision`, `torchaudio`
- `transformers` (HuggingFace)
- `mediapipe`, `opencv-python`
- `fastapi`, `uvicorn`
- `numpy`, `pandas`, `scipy`
- `hydra-core`, `omegaconf`
- `wandb`
