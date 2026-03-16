"""
ChoreoAI — Multimodal-to-Motion Translation Framework.

Converts text, image, and audio modalities into 3D skeletal dance sequences
using a shared latent embedding and a conditional diffusion generator.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("choreoai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0-dev"

# Public API
from choreoai.inference import generate_motion
from choreoai.encoders.motion_encoder import MotionEncoder
from choreoai.encoders.text_encoder import TextEncoder
from choreoai.generator.denoising_transformer import DenoisingTransformer

__all__ = [
    "generate_motion",
    "MotionEncoder",
    "TextEncoder",
    "DenoisingTransformer",
]
