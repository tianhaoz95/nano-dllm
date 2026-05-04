import torch
import torch.nn as nn
import torch.nn.functional as F


class MiCALinear(nn.Module):
    """
    Minor Component Adaptation (MiCA) wrapper for nn.Linear.

    For a frozen weight W, computes SVD W = UΣVᵀ and fixes
    B = U[:, -r:] (the r minor left singular vectors). Only
    A ∈ ℝ^(r × d_in) is trained, zero-initialized so ΔW = 0 at start.

    Forward: y = Wx + (α/r) · B·A·x
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        weight = base_layer.weight.data  # (d_out, d_in)
        dtype, device = weight.dtype, weight.device

        # SVD in FP32 for numerical stability; run on same device as weight
        U, _, _ = torch.linalg.svd(weight.float(), full_matrices=False)
        # U: (d_out, min(d_out, d_in)); singular values descend → last r are minor
        B = U[:, -rank:].to(dtype)  # (d_out, r)

        self.register_buffer("weight", weight)
        self.register_buffer("B", B)
        # Register bias as a buffer (None is valid and will be skipped by .to())
        bias_data = base_layer.bias.data.clone() if base_layer.bias is not None else None
        self.register_buffer("bias", bias_data)

        # Only A is trained; zero-init guarantees ΔW = 0 at the start of training
        self.A = nn.Parameter(
            torch.zeros(rank, weight.shape[1], dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        # x → (batch, r) → (batch, d_out), never materialising the full ΔW
        delta = F.linear(F.linear(x, self.A), self.B) * self.scaling
        return base + delta

    def extra_repr(self) -> str:
        d_out, d_in = self.weight.shape
        return (
            f"d_in={d_in}, d_out={d_out}, rank={self.rank}, alpha={self.alpha}"
        )
