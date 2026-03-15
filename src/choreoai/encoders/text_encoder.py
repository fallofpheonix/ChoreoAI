"""
text_encoder.py — Pre-trained RoBERTa text encoder with projection head.

Loads ``roberta-base`` from HuggingFace Transformers, freezes base weights,
and adds a trainable linear projection to the shared latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


class TextEncoder(nn.Module):
    """Frozen RoBERTa encoder with trainable projection head.

    Args:
        model_name: HuggingFace model identifier (default ``"roberta-base"``).
        latent_dim: Output dimensionality for the shared latent space.
        freeze_base: Whether to freeze all pre-trained weights.
        max_length: Maximum token sequence length.

    Example::

        encoder = TextEncoder(latent_dim=256)
        tokens = encoder.tokenize(["a dancer leaps across the stage"])
        z = encoder(**tokens)   # (1, 256)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        latent_dim: int = 256,
        freeze_base: bool = True,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        self.roberta = AutoModel.from_pretrained(model_name)

        if freeze_base:
            for param in self.roberta.parameters():
                param.requires_grad = False

        hidden_size: int = self.roberta.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Lazily load (and cache) the corresponding tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def tokenize(
        self,
        texts: list[str],
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Tokenise a list of strings and return input tensors.

        Args:
            texts: Batch of text strings.
            device: Optional target device.

        Returns:
            Dict with ``input_ids``, ``attention_mask`` (and optionally
            ``token_type_ids``) as tensors.
        """
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        return encoding

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """Encode a batch of tokenised texts.

        Args:
            input_ids: ``(B, L)`` token id tensor.
            attention_mask: ``(B, L)`` attention mask.
            token_type_ids: ``(B, L)`` token type ids (optional; RoBERTa
                does not use them but the argument is accepted for
                API compatibility).

        Returns:
            Latent embedding ``(B, latent_dim)``.
        """
        kwargs: dict[str, Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        outputs = self.roberta(**kwargs)

        # Use [CLS] token representation (first token)
        cls_rep: Tensor = outputs.last_hidden_state[:, 0, :]  # (B, hidden)

        return self.projection(cls_rep)  # (B, latent_dim)

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def encode_texts(
        self,
        texts: list[str],
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Tokenise and encode texts in a single call.

        Args:
            texts: List of raw text strings.
            device: Target device.

        Returns:
            ``(B, latent_dim)`` embedding tensor.
        """
        tokens = self.tokenize(texts, device=device)
        return self.forward(**tokens)
