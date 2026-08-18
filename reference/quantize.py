"""INT8 quantization scheme for the accelerator.

    STUB — implement in F0.3. See docs/ROADMAP.md.

This is the trickiest design task of Phase 0, and it's worth understanding why before writing a
single line of code.

## The problem

Multiplying matrices in INT8 is easy: you multiply int8 × int8, accumulate in int32, and rescale
at the end by the product of the scales. That covers QKᵀ and S·V without drama.

The problem is the **softmax in the middle**. `exp()` is not linear, so you can't "pull the scale
out" the way you can in a matrix multiply. There are three paths, and one has to be chosen
explicitly and documented in DECISIONS.md:

  (a) **Dequantize, softmax in floating point, requantize.**
      Easiest and most accurate, but it puts a floating-point unit in the datapath — expensive in
      DSPs and in area on an XC7Z020 that is already tight. It rules out the 16×16 target.

  (b) **Fixed point with a LUT for the exponential.**
      What real hardware does. A table in BRAM for exp() over a bounded range (after subtracting
      the row max, the argument is always <= 0, which bounds the range conveniently). Costs BRAM
      and precision, saves DSPs.

  (c) **Polynomial approximation, or exp2 with shifts.**
      Cheapest in area, hardest to get numerically right.

## What has to be implemented in F0.3

  1. `quantize_symmetric(x, num_bits=8)` — symmetric per-tensor scale, returning (int8, scale).
     Symmetric rather than asymmetric because it avoids the zero-point term in the accumulator,
     which in hardware is extra logic on the critical path for every MAC.

  2. `dequantize(x_int, scale)`.

  3. The chosen softmax path, implemented in Python in a way that is **bit-exactly simulable** in
     C++ and in Verilog. If the Python model uses float64 internally and the hardware uses 16-bit
     fixed point, they will not match and F0.10 becomes impossible to debug.

  4. An error report against FP32 (`attention_ref.attention_fp32`) over the three vector sets.
     That is where the **tolerance** used by F0.10's validation comes from. That tolerance has to
     be justified with data, not eyeballed upward until the tests pass.

## Warning

The temptation is to pick (a) to move fast and "fix it later in the RTL". Don't: if the Phase 0
model uses floating point and the Phase 1 RTL uses fixed point, the F1.7 co-simulation will show
discrepancies everywhere and you won't be able to tell a real bug from quantization noise. The
model has to be faithful to the hardware you plan to build — that fidelity is the entire project.
"""

from __future__ import annotations

import numpy as np


def quantize_symmetric(x: np.ndarray, num_bits: int = 8) -> tuple[np.ndarray, float]:
    """Quantizes to a signed integer using a symmetric per-tensor scale.

    Returns:
        (x_int, scale) such that x ≈ x_int * scale
    """
    raise NotImplementedError("F0.3 pending — see the module docstring.")


def dequantize(x_int: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of `quantize_symmetric`."""
    raise NotImplementedError("F0.3 pending — see the module docstring.")


def quantized_attention(
    q_int: np.ndarray,
    k_int: np.ndarray,
    v_int: np.ndarray,
    scales: dict[str, float],
) -> np.ndarray:
    """Full INT8 attention, faithful to the datapath that will be built in hardware.

    This is the golden output the C++ simulator consumes.
    """
    raise NotImplementedError("F0.3 pending — decide the softmax path first.")
