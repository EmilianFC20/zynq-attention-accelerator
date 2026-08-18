"""FP32 golden reference for attention.

    STUB — implement in F0.2. See docs/ROADMAP.md.

This is the definition of "correct" for the whole project. Everything else — the C++ model, the
RTL, the accelerator on the Zynq — is validated against what this file produces.

What has to be implemented in F0.2:

  1. `attention_fp32(q, k, v)`: single-head attention, written by hand with basic torch
     operations (matmul, softmax, scaling). Do NOT use
     `torch.nn.functional.scaled_dot_product_attention` in the implementation — it is used only
     to *validate* that the hand-written implementation matches. The point is to have every
     intermediate step accessible, because the C++ model will need to compare against the
     intermediates (the S matrix before and after softmax), not just against the final output.

  2. Return the intermediates: `scores` (QKᵀ/√d) and `probs` (after softmax). When the F0.9 flash
     dataflow produces a different result, you will need to know at which stage it diverged —
     without the intermediates that debugging is blind.

  3. Fixed seed and determinism. Two runs must give bit-identical results.

A note on why operation order matters: the stable softmax subtracts the row max before
exponentiating. The *online* softmax in the flash dataflow (F0.9) does exactly that, but
incrementally, rescaling the partial accumulators as a new max appears. If this reference doesn't
implement the stable version, the comparison against flash will show differences that look like
accelerator bugs when they are really reference bugs.
"""

from __future__ import annotations

import torch


def attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-head attention in FP32.

    Args:
        q: (N, d) float32
        k: (N, d) float32
        v: (N, d) float32

    Returns:
        (out, scores, probs) where:
            out    (N, d) — the attention output
            scores (N, N) — QKᵀ / sqrt(d), before softmax
            probs  (N, N) — after the row-wise stable softmax
    """
    raise NotImplementedError(
        "F0.2 pending — see docs/ROADMAP.md for the scope of this task."
    )


def validate_against_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    """Compares `attention_fp32` against PyTorch's reference implementation.

    Returns:
        The maximum absolute difference between the two outputs.
    """
    raise NotImplementedError(
        "F0.2 pending — use torch.nn.functional.scaled_dot_product_attention as the oracle."
    )
