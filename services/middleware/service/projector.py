"""Audio projector: maps MedASR encoder embeddings into MedGemma's embedding space.

Architecture:
  Input:  [batch, seq_len, 512]    (MedASR Conformer encoder output)
    | Frame stacking (k=4): concat 4 adjacent frames
  [batch, seq_len/4, 2048]
    | Linear(2048 -> 2560, no bias)
    | RMSNorm(2560)
    | GELU
    | Linear(2560 -> 2560, no bias)
  [batch, seq_len/4, 2560]
    | Prepend <audio_start>, append <audio_end> learned delimiters
  Output: [batch, seq_len/4 + 2, 2560]  (MedGemma embedding space)

The <audio_start>/<audio_end> delimiters give the frozen LLM a clean signal
that the embedding type has changed from text to projected audio and back.
Without these, the LLM has no way to distinguish audio embeddings from text
embeddings since the projector is explicitly trained to make them look similar.

~11.8M + 5,120 trainable parameters.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class AudioProjector(nn.Module):
    """Projects MedASR encoder embeddings into MedGemma's embedding space.

    Args:
        encoder_dim: MedASR encoder hidden size (default: 512).
        llm_dim: MedGemma hidden size (default: 2560).
        stack_factor: Number of adjacent frames to concatenate (default: 4).
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 2560,
        stack_factor: int = 4,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.stack_factor = stack_factor

        stacked_dim = encoder_dim * stack_factor

        self.proj = nn.Sequential(
            nn.Linear(stacked_dim, llm_dim, bias=False),
            RMSNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False),
        )

        self.audio_start = nn.Parameter(torch.randn(llm_dim) * 0.02)
        self.audio_end = nn.Parameter(torch.randn(llm_dim) * 0.02)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Project encoder embeddings to LLM embedding space.

        Args:
            encoder_output: [batch, seq_len, encoder_dim] from MedASR encoder.

        Returns:
            [batch, seq_len // stack_factor + 2, llm_dim] —
            <audio_start> + projected frames + <audio_end>.
        """
        B, T, D = encoder_output.shape
        assert D == self.encoder_dim, (
            f"Expected encoder_dim={self.encoder_dim}, got {D}"
        )

        k = self.stack_factor

        remainder = T % k
        if remainder != 0:
            pad_len = k - remainder
            padding = encoder_output.new_zeros(B, pad_len, D)
            encoder_output = torch.cat([encoder_output, padding], dim=1)
            T = encoder_output.shape[1]

        stacked = encoder_output.reshape(B, T // k, D * k)
        projected = self.proj(stacked)  # [B, T/k, llm_dim]

        start = self.audio_start.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        end = self.audio_end.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        return torch.cat([start, projected, end], dim=1)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
