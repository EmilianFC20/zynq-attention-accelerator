"""The INT8 attention pipeline: the arithmetic of `quantize.py` assembled end to end.

`quantize.py` specifies the operations; this file specifies the *order* they run in. Together they
are what the C++ model of Phase 0 and the RTL of Phase 1 have to reproduce bit for bit (D-007).

Nothing here does floating-point arithmetic on tensor data. Floats appear at three points, all of
which a real system performs once on the ARM core before the accelerator starts:

  1. quantizing Q, K, V and recording their scales,
  2. folding those scales into the integer multiplier that feeds the exponent path,
  3. dequantizing the output for comparison against FP32.

Everything between (2) and (3) is integers, and `attention_int8` returns those integers so the
comparison can be made at whichever stage a divergence needs to be localized to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from attention_ref import attention_fp32
from quantize import (
    OUT_FRAC_BITS,
    derive_score_multiplier,
    dequantize,
    floor_div,
    int_softmax,
    quantize_symmetric,
)


@dataclass
class Int8Attention:
    """Everything the INT8 pipeline produced, at every stage.

    Carrying the intermediates rather than only `out` is the same choice `attention_fp32` makes and
    for the same reason: when the C++ model disagrees, `scores_int` versus `probs_uint8` versus
    `out_int` says *which unit* is wrong. A mismatch on the final output alone says only that one
    of four stages is.
    """

    out: np.ndarray             # (N, d) float64 — dequantized, for comparison against FP32
    out_int: np.ndarray         # (N, d) int    — the integers the accelerator actually emits
    out_scale: float            # real value of one out_int LSB
    scores_int: np.ndarray      # (N, N) int32  — exact q8 @ k8^T
    probs_uint8: np.ndarray     # (N, N) uint8  — unnormalized exponentials, D-006
    row_sums: np.ndarray        # (N, 1) int    — l = sum p8, the softmax denominators
    scale_q: float
    scale_k: float
    scale_v: float


def attention_int8(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> Int8Attention:
    """Single-head attention with an INT8 datapath and an INT32 accumulator.

    Args:
        q, k, v: (N, d) real-valued arrays (or torch tensors).

    Returns:
        An `Int8Attention` holding the output and every intermediate.
    """
    q, k, v = (np.asarray(t.numpy() if isinstance(t, torch.Tensor) else t, dtype=np.float64)
               for t in (q, k, v))

    if not (q.shape == k.shape == v.shape) or q.ndim != 2:
        raise ValueError(f"q, k, v must be matching 2-D (N, d) arrays: "
                         f"{q.shape}, {k.shape}, {v.shape}")

    head_dim = q.shape[1]

    q8, scale_q = quantize_symmetric(q)
    k8, scale_k = quantize_symmetric(k)
    v8, scale_v = quantize_symmetric(v)

    # Stage 1 — S = q8 @ k8^T. Exact: with d = 64, the largest possible magnitude is
    # 64 * 127 * 127 = 1_032_256, three orders of magnitude inside INT32. The 1/sqrt(d) factor is
    # deliberately *not* applied here; it rides along in the score multiplier below.
    scores_int = q8 @ k8.T

    # Stage 2 — integer softmax. The one place a scale has to be pushed inside the arithmetic,
    # because exp() is not linear and will not let a constant be factored back out.
    score_mult, score_shift = derive_score_multiplier(scale_q, scale_k, head_dim)
    probs_uint8, row_sums = int_softmax(scores_int, score_mult, score_shift)

    # Stage 3 — O_acc = p8 @ v8, again exact, and the row sums from stage 2 normalize it. The
    # numerator is left-shifted first so the division keeps OUT_FRAC_BITS of fraction (D-009): the
    # quotient is a weighted *average* of v, whose magnitude shrinks as N grows, so rounding it
    # onto v's own grid would throw away most of the signal.
    out_acc = probs_uint8 @ v8
    out_int = floor_div(out_acc << OUT_FRAC_BITS, row_sums)
    out_scale = scale_v / (1 << OUT_FRAC_BITS)

    return Int8Attention(
        out=dequantize(out_int, out_scale),
        out_int=out_int,
        out_scale=out_scale,
        scores_int=scores_int,
        probs_uint8=probs_uint8,
        row_sums=row_sums,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )


@dataclass
class QuantizationError:
    """How far the INT8 pipeline lands from FP32, for one (N, d)."""

    seq_len: int
    head_dim: int
    max_abs: float          # worst single-element error
    rms: float              # root mean square error over all N*d outputs
    reference_std: float    # spread of the FP32 output, so `rms` can be read as a fraction
    relative_rms: float     # rms / reference_std — the number that is comparable across N

    def __str__(self) -> str:
        return (f"N={self.seq_len:<5} d={self.head_dim:<3} "
                f"max|err|={self.max_abs:.3e}  rms={self.rms:.3e}  "
                f"rms/sigma={self.relative_rms:.4f}")


def quantization_error(seq_len: int, head_dim: int, seed: int = 0xC0FFEE) -> QuantizationError:
    """Measures the INT8 pipeline against the FP32 reference on deterministic inputs.

    This is an *accuracy* measurement, not a pass/fail check (D-007): the two pipelines compute
    genuinely different things, and the gap between them is a property of the design to report,
    not a threshold to clear. The bit-exactness thresholds live on the C++ and RTL side, where
    both sides are computing the same integer function and any difference at all is a bug.

    `relative_rms` is reported because the absolute error is not comparable across N: attention
    averages over N values, so the output's own magnitude shrinks as N grows and a constant
    absolute error would look like an improving one.
    """
    from attention_ref import make_inputs

    q, k, v = make_inputs(seq_len, head_dim, seed=seed)
    reference = attention_fp32(q, k, v)[0].numpy().astype(np.float64)
    result = attention_int8(q, k, v)

    difference = result.out - reference
    reference_std = float(reference.std())

    rms = float(np.sqrt((difference ** 2).mean()))
    return QuantizationError(
        seq_len=seq_len,
        head_dim=head_dim,
        max_abs=float(np.abs(difference).max()),
        rms=rms,
        reference_std=reference_std,
        relative_rms=rms / reference_std,
    )


if __name__ == "__main__":
    print("INT8 pipeline vs FP32 reference\n")
    for n, d in [(8, 8), (32, 16), (128, 64), (256, 64), (512, 64)]:
        print(f"  {quantization_error(n, d)}")
