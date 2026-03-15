"""
diffusion_scheduler.py — DDPM noise schedule and sampling utilities.

Implements the Denoising Diffusion Probabilistic Model (DDPM) noise schedule
from Ho et al. (2020), providing:

* :class:`DDPMScheduler` — registers β-schedule buffers, computes forward
  diffusion, and performs single reverse-denoising steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DDPMScheduler(nn.Module):
    """DDPM noise scheduler for motion diffusion.

    Implements the linear (or cosine) variance schedule from
    ``Ho et al. (2020) — Denoising Diffusion Probabilistic Models``.

    Args:
        num_timesteps: Total diffusion timesteps ``T``.
        beta_start: Starting noise variance (β₁).
        beta_end: Ending noise variance (β_T).
        schedule: ``"linear"`` or ``"cosine"`` variance schedule.
        clip_sample: Whether to clip predicted x₀ to ``[-1, 1]``.

    Example::

        scheduler = DDPMScheduler(num_timesteps=1000)
        x0 = torch.randn(4, 120, 17, 3)
        t  = torch.randint(0, 1000, (4,))
        x_t, noise = scheduler.forward_diffusion(x0, t)
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        clip_sample: bool = True,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps
        self.clip_sample = clip_sample

        betas = self._build_schedule(num_timesteps, beta_start, beta_end, schedule)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )

        # Register all quantities as buffers (auto-device movement)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Schedule builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_schedule(
        T: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
    ) -> Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule
            steps = T + 1
            x = torch.linspace(0, T, steps) / T
            f = torch.cos((x + 0.008) / 1.008 * torch.pi / 2) ** 2
            f = f / f[0]
            betas = 1 - f[1:] / f[:-1]
            return betas.clamp(0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}. Choose 'linear' or 'cosine'.")

    # ------------------------------------------------------------------
    # Helper: gather scalar per sample from a 1-D buffer
    # ------------------------------------------------------------------

    def _gather(self, coeff: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
        """Gather per-timestep coefficients and reshape to broadcast.

        Args:
            coeff: 1-D buffer of length ``T``.
            t: ``(B,)`` timestep indices.
            shape: Shape of the target tensor to broadcast against.

        Returns:
            Tensor of shape broadcastable to *shape*.
        """
        out = coeff.gather(0, t)                       # (B,)
        # Append singleton dims for broadcasting: (B, 1, 1, …)
        for _ in range(len(shape) - 1):
            out = out.unsqueeze(-1)
        return out

    # ------------------------------------------------------------------
    # Forward diffusion  q(x_t | x_0)
    # ------------------------------------------------------------------

    def forward_diffusion(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample ``x_t`` given clean data ``x_0`` and timestep ``t``.

        Uses the closed-form:
            ``x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε``

        Args:
            x_0: Clean data tensor of arbitrary shape ``(B, ...)``.
            t: ``(B,)`` integer timesteps.
            noise: Optional pre-sampled noise (same shape as ``x_0``).
                   If ``None``, standard Gaussian noise is used.

        Returns:
            Tuple ``(x_t, noise)`` — the noised sample and the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._gather(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_t = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_t * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse step  p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        model_output: Tensor,
        t: Tensor,
        x_t: Tensor,
        add_noise: bool = True,
    ) -> Tensor:
        """Perform a single DDPM reverse denoising step.

        Given the predicted noise ``ε_θ(x_t, t)``, computes
        ``x_{t-1} ~ p_θ(x_{t-1} | x_t)``.

        Args:
            model_output: Predicted noise ``(B, ...)`` — same shape as ``x_t``.
            t: ``(B,)`` integer timesteps.
            x_t: Current noisy tensor ``(B, ...)``.
            add_noise: Whether to add stochastic noise (set ``False`` for the
                       last denoising step ``t=0``).

        Returns:
            ``x_{t-1}`` tensor of the same shape as ``x_t``.
        """
        # Predict x_0 from ε-prediction
        sqrt_recip_alphas_t = self._gather(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_t = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        betas_t = self._gather(self.betas, t, x_t.shape)

        # Mean of posterior q(x_{t-1} | x_t, x_0)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_t * model_output)

        if self.clip_sample:
            model_mean = model_mean.clamp(-1.0, 1.0)

        if add_noise and (t > 0).any():
            posterior_var_t = self._gather(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            # Only add noise for t > 0
            t_broadcast = t.reshape(-1, *([1] * (x_t.dim() - 1))).float()
            model_mean = model_mean + (t_broadcast > 0).float() * posterior_var_t.sqrt() * noise

        return model_mean

    # ------------------------------------------------------------------
    # Full reverse sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        conditioning: Tensor,
        device: torch.device | str = "cpu",
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the full DDPM reverse diffusion sampling loop.

        Args:
            model: The denoising model (e.g. :class:`DenoisingTransformer`).
                   Signature: ``model(x_t, t, z) → ε``.
            shape: Shape of the output tensor ``(B, T, K, 3)`` or similar.
            conditioning: ``(B, latent_dim)`` conditioning embedding.
            device: Target device.
            key_padding_mask: Optional padding mask ``(B, T)``.

        Returns:
            Generated sample of the given *shape*.
        """
        x = torch.randn(*shape, device=device)  # start from pure noise

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            noise_pred = model(x, t, conditioning, key_padding_mask)
            add_noise = i > 0
            x = self.reverse_step(noise_pred, t, x, add_noise=add_noise)

        return x
