"""FP32 golden reference for attention.

This is the definition of "correct" for the whole project. Everything else — the C++ model, the
RTL, the accelerator on the Zynq — is validated against what this file produces.

The implementation is deliberately hand-written out of basic torch operations rather than calling
`torch.nn.functional.scaled_dot_product_attention`. Two reasons:

  1. SDPA returns only the final (N, d) output and discards the intermediates. The blocked
     dataflow needs `scores` (QK^T / sqrt(d)) and `probs` (after softmax) to localize a
     divergence: without them, a mismatch tells you only *that* the result is wrong, never *where*
     it went wrong.
  2. Implementing with SDPA and validating against SDPA would be checking a function against
     itself. Hand-writing it and then comparing gives two independent paths to the same number.

SDPA is therefore used only as the oracle, in `validate_against_torch`.

A note on why operation order matters: the stable softmax subtracts the row max before
exponentiating. The *online* softmax in the blocked dataflow does exactly that, but incrementally,
rescaling the partial accumulators as a new max appears. If this reference did not implement the
stable version, the comparison against the blocked dataflow would show differences that look like
accelerator bugs when they are really reference bugs.
"""

from __future__ import annotations

import math

import torch


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Row-wise numerically stable softmax.

    Computes, for every row independently:

        softmax(s)_i = exp(s_i - m) / sum_j exp(s_j - m),   with m = max_j(s_j)

    Subtracting the row max is algebraically a no-op (the exp(-m) factor cancels between the
    numerator and the denominator) but numerically essential: FP32 exp() overflows to +inf around
    88, and a single inf turns the whole row into NaN through inf/inf.

    Args:
        scores: (N, N) float32. Softmax is applied along the last dimension, so each output row
            is a probability distribution over the N keys.

    Returns:
        (N, N) float32, every row non-negative and summing to 1.
    """
    # TODO(human): implement the stable softmax.
    #
    # Useful pieces:
    #   scores.max(dim=-1, keepdim=True).values  -> (N, 1) row maxima
    #   torch.exp(x)                             -> elementwise exponential
    #   x.sum(dim=-1, keepdim=True)              -> (N, 1) row sums
    #
    # keepdim=True matters: without it the reduction returns shape (N,), which broadcasts against
    # (N, N) along the wrong axis and silently normalizes columns instead of rows.
    raise NotImplementedError("stable_softmax is not implemented yet")


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
            scores (N, N) — QK^T / sqrt(d), before softmax
            probs  (N, N) — after the row-wise stable softmax
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(f"expected 2-D (N, d) tensors, got {q.ndim}/{k.ndim}/{v.ndim} dims")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q, k, v must share a shape: {tuple(q.shape)}, "
                         f"{tuple(k.shape)}, {tuple(v.shape)}")
    if q.dtype != torch.float32:
        raise ValueError(f"this reference is FP32 only, got {q.dtype}")

    head_dim = q.shape[1]

    # The scale is applied *after* the matmul rather than folded into q beforehand. The two are
    # equivalent in FP32, but not once the datapath is INT8: scaling q first would change what
    # lands in the INT32 accumulator. Keeping it here leaves the accumulator exact and pushes the
    # scaling into requantization, which is where the hardware performs it anyway.
    scores = (q @ k.transpose(0, 1)) / math.sqrt(head_dim)
    probs = stable_softmax(scores)
    out = probs @ v

    return out, scores, probs


def validate_against_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    """Compares `attention_fp32` against PyTorch's reference implementation.

    SDPA expects a leading batch/head layout of (batch, heads, N, d), so the 2-D single-head
    tensors are unsqueezed and squeezed back around the call.

    The result will not be bit-identical: SDPA dispatches to a fused kernel whose accumulation
    order differs, and FP32 addition is not associative. A difference on the order of 1e-6 is the
    expected outcome, not a bug — which is why this returns a magnitude for the caller to compare
    against a tolerance instead of asserting equality.

    Returns:
        The maximum absolute difference between the two outputs.
    """
    ours, _, _ = attention_fp32(q, k, v)

    reference = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).unsqueeze(0),
        k.unsqueeze(0).unsqueeze(0),
        v.unsqueeze(0).unsqueeze(0),
    ).squeeze(0).squeeze(0)

    return (ours - reference).abs().max().item()


def make_inputs(
    seq_len: int,
    head_dim: int,
    seed: int = 0xC0FFEE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic FP32 (N, d) inputs for tests and experiments.

    An explicit generator is used instead of `torch.manual_seed` so that calling this never
    perturbs the caller's global RNG state: two runs give bit-identical tensors regardless of what
    else the process did in between.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (seq_len, head_dim)

    def sample() -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    return sample(), sample(), sample()
