"""
evaluate_model.py — Standalone evaluation suite for ChoreoAI models.
"""

import torch
import numpy as np
from pathlib import Path
from choreoai.evaluate import compute_fmd, extract_motion_features, retrieval_accuracy
from choreoai.encoders.text_encoder import TextEncoder
from choreoai.encoders.motion_encoder import MotionEncoder

def compute_diversity(features: torch.Tensor) -> float:
    """Compute average pairwise distance between feature vectors."""
    if features.size(0) < 2: return 0.0
    dist = torch.cdist(features, features)
    # Average upper triangle
    n = features.size(0)
    avg_dist = dist.sum() / (n * (n - 1))
    return avg_dist.item()

def compute_smoothness(motion: torch.Tensor) -> float:
    """Average acceleration norm across all joints."""
    # motion: (B, T, K, 3)
    accel = motion[:, 2:] - 2 * motion[:, 1:-1] + motion[:, :-2]
    norm = torch.linalg.norm(accel, dim=-1).mean()
    return norm.item()

def compute_stability(motion: torch.Tensor) -> float:
    """Check bone length variance as a proxy for skeleton stability."""
    # This is a stub: real implementation would check variance of distances
    # between connected joints according to a skeleton map.
    return 0.0

def run_evaluation(gen_motions, real_motions, text_prompts, device="cpu"):
    """Run full evaluation suite."""
    m_encoder = MotionEncoder(latent_dim=256).to(device)
    t_encoder = TextEncoder(latent_dim=256).to(device)
    
    gen_feats = extract_motion_features(gen_motions, m_encoder, device=device)
    real_feats = extract_motion_features(real_motions, m_encoder, device=device)
    
    text_feats = t_encoder.encode_texts(text_prompts, device=device)
    
    results = {
        "FMD": compute_fmd(real_feats, gen_feats),
        "Diversity": compute_diversity(gen_feats),
        "Smoothness": compute_smoothness(torch.stack(gen_motions)),
        "Alignment": retrieval_accuracy(text_feats, gen_feats, top_k=[1, 5, 10]),
        "Stability": compute_stability(torch.stack(gen_motions))
    }
    
    return results

if __name__ == "__main__":
    # Placeholder for running from CLI
    pass
