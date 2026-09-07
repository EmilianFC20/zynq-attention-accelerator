"""INT8 quantization scheme for the accelerator.

This module is the *specification* of the accelerator's arithmetic, written in Python. Every
operation here is an integer operation that a DSP48E1 and a handful of LUTs can perform. There is
no floating point anywhere in the datapath: floats appear only at the two boundaries, where real
tensors are turned into integers (`quantize_symmetric`) and where the design's real-valued
constants are turned into an integer multiplier and a shift (`derive_multiplier`).

That constraint is not stylistic. `docs/DECISIONS.md` D-007 asks the C++ model and later the RTL to
reproduce this file *bit for bit*, which is only possible if this file never does anything they
cannot do.

## The shape of the pipeline

The rule the whole design follows is: **narrow at the boundaries, wide in the middle.** Every
tensor that is stored or transmitted is 8 bits; every accumulation that happens inside a unit is
32 bits and exact.

    Q,K,V fp32  --quantize-->  q8,k8,v8 int8 + scales s_q,s_k,s_v
                S_i32 = q8 @ k8^T                        exact, real value = S_i32 * s_q*s_k
                p8    = int_softmax(S_i32)               requantized to uint8   (D-004, D-006)
                O_acc = p8 @ v8,  l = sum p8             exact
                O_q   = (O_acc << 8) // l                requantized, D-009
                O     = O_q * (s_v / 256)  --dequantize-->  fp32

A matrix multiply lets the scale be factored out of the integer arithmetic entirely:
`(a*s_a)(b*s_b) = (a*b)*s_a*s_b`. The exponential does not — it is nonlinear, so the scale has to be
pushed *inside* it. That is the reason `derive_multiplier` exists and the reason softmax is the
hard part of this phase.

## Where the error actually comes from

Not from the matmuls: an INT32 accumulator holds `64 * 127 * 127` with room to spare, so `q8 @ k8^T`
is the exact integer dot product, not an approximation of it. All of the quantization error is
introduced at three specific points, and F0.3 measures each of them:

  1. Rounding Q, K, V onto the INT8 grid (`quantize_symmetric`).
  2. The 256-entry `exp2` table and the fixed-point exponent path (D-004).
  3. Requantizing the exponentials to UINT8 (D-006).
"""

from __future__ import annotations

import math

import numpy as np

# Widths of the design, in one place so that a change here is a change everywhere.
INT8_MAX = 127          # symmetric range: [-127, 127]. -128 is deliberately unused, see below.
UINT8_MAX = 255         # probabilities, D-006
MULT_BITS = 15          # bit width of the requantization multiplier M, see derive_multiplier


def round_shift(x: np.ndarray | int, s: int) -> np.ndarray | int:
    """Arithmetic right shift by `s` with round-half-up. The only rounding rule in the design.

    D-005: every fixed-point right shift in the Python reference, the C++ model and the Verilog
    rounds by adding a half-LSB before shifting. Truncation would bias every term in the same
    direction, and that bias accumulates linearly with N while random rounding error grows only as
    sqrt(N).

    Numpy's `>>` on a signed integer array is an arithmetic shift (it floors toward -inf), which is
    what C++20 mandates for signed types and what Verilog's `>>>` does on a `signed` operand. The
    three languages therefore agree by construction.

    `s` may be an array as well as a scalar. The exponent path of D-004 shifts each element by a
    different amount -- one barrel shifter fed by the integer part of the exponent -- so a
    per-element shift is the datapath's real behaviour, not a convenience for NumPy.

    Args:
        x: signed integer scalar or array.
        s: shift amount, >= 1 (a half-LSB is meaningless at s = 0). Scalar or array.
    """
    if np.any(np.asarray(s) < 1):
        raise ValueError(f"round_shift needs s >= 1, got {s}")
    return (x + (1 << (s - 1))) >> s


def floor_div(numerator: np.ndarray | int, denominator: np.ndarray | int) -> np.ndarray | int:
    """Division that floors toward -inf, for both signs of the numerator.

    This exists because it is the one place D-005's shift rule cannot reach: the final
    normalization `O = O_acc / l` is a true division, and `O_acc` is signed because V is signed.
    Python's `//` floors while C++'s `/` truncates toward zero, so `-7 / 2` is -4 in Python and -3
    in C++. Leaving that implicit would produce a one-LSB disagreement on roughly half the negative
    outputs -- exactly the kind of mismatch that is cheap to prevent now and miserable to find
    during F1.7 co-simulation.

    The C++ and Verilog sides must implement floor semantics explicitly to match this.
    """
    return numerator // denominator


def choose_scale(x: np.ndarray, num_bits: int = 8) -> float:
    """Picks the per-tensor scale that maps a real tensor onto the signed integer grid.

    The scale is the single number that defines the quantization of a tensor: `x ~= x_int * scale`.
    Choosing it is *calibration*. D-008: the statistic is the **absolute maximum**, so nothing ever
    clips and the only error introduced here is the half-LSB of `rint`.

    The alternative worth naming is percentile clipping (99.9%, say), which deliberately saturates
    the tail to buy a finer grid for the bulk. That trade pays when a distribution has a long, thin
    tail wasting the range -- the outlier channels that motivate LLM.int8() and SmoothQuant in real
    transformers. It does not pay here: Q, K and V are Gaussian by construction, the sample maximum
    sits at a perfectly ordinary ~4.7 sigma, and clipping it away costs more than the half-bit of
    resolution it returns. The measurement is in D-008.

    Args:
        x: the real-valued tensor to calibrate against.
        num_bits: width of the signed integer format. 8 gives a usable range of [-127, 127].

    Returns:
        A strictly positive float scale.
    """
    qmax = (1 << (num_bits - 1)) - 1     # 127 for num_bits=8

    absmax = float(np.abs(np.asarray(x, dtype=np.float64)).max())

    # An all-zero tensor has no scale that means anything. Returning the smallest representable
    # step keeps `x_int` at zero (the correct answer) without making the division a NaN.
    if absmax == 0.0:
        return 1.0 / qmax

    return absmax / qmax


def quantize_symmetric(x: np.ndarray, num_bits: int = 8) -> tuple[np.ndarray, float]:
    """Quantizes to a signed integer using a symmetric per-tensor scale.

    Symmetric (no zero point) rather than affine because an affine quantization `x ~= (q - z) * s`
    expands a dot product into `sum(q_a*q_b) - z_a*sum(q_b) - z_b*sum(q_a) + N*z_a*z_b`. The three
    correction terms are cheap on a CPU and genuinely annoying in a systolic array: they need extra
    accumulators and extra logic hanging off the critical path of every PE. Zero-centred Gaussian
    activations lose almost nothing to symmetry, so the asymmetric variant buys accuracy the design
    does not need at a price the fabric cannot afford.

    Per-tensor rather than per-row because a per-row scale would make the score scale
    `s_q[i] * s_k[j]` a function of both indices, so the requantization multiplier feeding softmax
    would change per element instead of being a single constant folded once into the pipeline.

    Returns:
        (x_int, scale) with `x_int` int8-valued (held in int64) such that `x ~= x_int * scale`.
    """
    scale = choose_scale(x, num_bits)
    qmax = (1 << (num_bits - 1)) - 1

    x_int = np.clip(np.rint(np.asarray(x, dtype=np.float64) / scale), -qmax, qmax)
    return x_int.astype(np.int64), float(scale)


def dequantize(x_int: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of `quantize_symmetric`. Host-side only -- the accelerator never does this."""
    return np.asarray(x_int, dtype=np.float64) * scale


def derive_multiplier(scale: float, mult_bits: int = MULT_BITS) -> tuple[int, int]:
    """Turns a real constant into an integer multiplier and a right shift: `scale ~= M >> s`.

    This is the idiom that lets a design with no floating-point hardware apply an arbitrary real
    constant. `math.frexp` splits `scale = m * 2^e` with the mantissa `m` in [0.5, 1); `m` is then
    rounded onto a `mult_bits`-wide integer grid and the exponent is absorbed into the shift. The
    result is used as `round_shift(x * M, s)`.

    `mult_bits` is 15, not the 31 that gemmlowp and TFLite use. Their target is a CPU with a 32x32
    multiplier, where the extra bits are free. The target here is a DSP48E1 whose multiplier is
    25x18 (D-003), so a 15-bit multiplicand fits one port with a bit to spare. The cost is a
    relative error of 2^-15 ~= 3e-5, which sits two orders of magnitude below the 1/255 ~= 3.9e-3
    resolution of the UINT8 probabilities (D-006) -- so the extra 16 bits of multiplier would buy
    exactly nothing measurable, in exchange for a wider multiplier in every PE.

    Returns:
        (M, s) with M in [2^(mult_bits-1), 2^mult_bits) and s >= 1.
    """
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ValueError(f"derive_multiplier needs a finite positive scale, got {scale}")

    mantissa, exponent = math.frexp(scale)          # scale == mantissa * 2**exponent, m in [0.5,1)
    m_int = int(round(mantissa * (1 << mult_bits)))

    # Rounding can push the mantissa up to exactly 1.0, which no longer fits in `mult_bits`.
    # Renormalize instead of saturating, so the represented value stays correct.
    if m_int == (1 << mult_bits):
        m_int >>= 1
        exponent += 1

    shift = mult_bits - exponent
    if shift < 1:
        raise ValueError(
            f"scale {scale} is too large for a {mult_bits}-bit multiplier "
            f"(would need shift {shift}); the datapath assumes fractional constants"
        )
    return m_int, shift


def apply_multiplier(x: np.ndarray | int, multiplier: int, shift: int) -> np.ndarray | int:
    """Applies a `(M, s)` pair from `derive_multiplier`: computes `round(x * scale)` in integers.

    Kept as a named function rather than inlined because it is the exact operation the RTL will
    instantiate, and naming it makes the model-to-RTL correspondence obvious at every call site.
    """
    return round_shift(np.asarray(x, dtype=np.int64) * multiplier, shift)


# ---------------------------------------------------------------------------------------------
# Integer softmax (D-004)
# ---------------------------------------------------------------------------------------------

LOG2E = 1.4426950408889634      # log2(e); converts a natural exponent into a base-2 one
EXP_FRAC_BITS = 8               # fractional bits of the exponent → 256-entry table, D-004
EXP_TABLE_BITS = 15             # table entries are UQ1.15, so 1.0 is 32768 and fits in 16 bits
EXP_MAX_SHIFT = 16              # 2^-16 is already below UINT8 resolution; clamp instead of shift
OUT_FRAC_BITS = 8               # extra fractional bits kept through the final division, D-009


def build_exp2_table() -> np.ndarray:
    """The 256-entry table of `2^(-f/256)` in UQ1.15, `f = 0..255` (D-004).

    Built at import time rather than written out as a literal so the derivation stays visible; the
    C++ model and the RTL will hold the same 512 bytes as a constant array / LUTRAM initializer,
    and `test_quantize.py` checks they would agree.

    Range is `(16384, 32768]`: entry 0 is exactly 1.0 and entry 255 is just above 0.5. Nothing in
    the table needs 17 bits, which is the whole reason for UQ1.15 over UQ0.16.
    """
    f = np.arange(1 << EXP_FRAC_BITS, dtype=np.float64)
    return np.rint(np.exp2(-f / (1 << EXP_FRAC_BITS)) * (1 << EXP_TABLE_BITS)).astype(np.int64)


EXP2_LUT = build_exp2_table()


def derive_score_multiplier(
    scale_q: float,
    scale_k: float,
    head_dim: int,
    mult_bits: int = MULT_BITS,
) -> tuple[int, int]:
    """The `(M, s)` pair that carries an INT32 score into the exponent domain.

    Everything the exponent path needs to know about the outside world is folded into one constant:

        C = (s_q * s_k / sqrt(d)) * log2(e) * 2^EXP_FRAC_BITS
              \\_______________/    \\______/   \\______________/
               score_scale          nat→base2   into Q8 fixed point

    `score_scale` turns an INT32 score back into a real one, `log2(e)` converts `exp(x)` into
    `2^(x·log2 e)` so the result can be split into a shift and a table lookup, and the last factor
    places it in the 8-fractional-bit format the table is indexed by. All three are compile-time
    constants, so folding them costs nothing at run time -- the hardware sees one multiplier and
    one shift, which is exactly what D-004 asked for.

    Note that `1/sqrt(d)` lands *here* rather than being applied to Q beforehand: keeping it out of
    the datapath is what leaves the INT32 accumulator exact (see `attention_ref.attention_fp32`).
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")

    score_scale = scale_q * scale_k / math.sqrt(head_dim)
    return derive_multiplier(score_scale * LOG2E * (1 << EXP_FRAC_BITS), mult_bits)


def int_exp2_neg(y: np.ndarray) -> np.ndarray:
    """Evaluates `2^(y / 256)` for `y <= 0`, returning UQ1.15, using only shifts and a lookup.

    With `-y = 256*n + f`, the identity `2^(-(256n+f)/256) = 2^-n · 2^(-f/256)` splits the problem
    into an exact barrel shift and a table lookup over a domain of width 1. The row-max subtraction
    that makes softmax numerically stable is what guarantees `y <= 0`, so a table covering a single
    octave covers the entire input range -- the stability trick and the cheap-hardware trick are
    the same trick.

    Two details that have to be explicit rather than implied by the shift:

      - `n >= 16` returns 0 outright. `2^-16 ~= 1.5e-5` is already an order of magnitude under the
        `1/255` step the result is about to be requantized to, so nothing is lost; and in C++ a
        shift of a 32-bit value by >= 32 is undefined behavior, which would surface as an
        unreproducible model-vs-RTL mismatch rather than an honest wrong answer.
      - the shift by `n` rounds (D-005). D-004 wrote it as a bare `>>`; rounding is the reading
        that keeps the "no site truncates" rule intact, and it costs one adder.
    """
    y = np.asarray(y, dtype=np.int64)
    if np.any(y > 0):
        raise ValueError("int_exp2_neg expects y <= 0; subtract the row max first")

    magnitude = -y
    n = magnitude >> EXP_FRAC_BITS                          # integer part → shift amount
    f = magnitude & ((1 << EXP_FRAC_BITS) - 1)              # fractional part → table index

    entry = EXP2_LUT[f]
    shifted = np.where(n == 0, entry, round_shift(entry, np.maximum(n, 1)))
    return np.where(n >= EXP_MAX_SHIFT, 0, shifted)


def int_softmax(
    scores_int: np.ndarray,
    score_multiplier: int,
    score_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise softmax over INT32 scores, in integers only, producing UINT8 probabilities.

    The pipeline, per row (D-004 and D-006):

        y  = (S - rowmax) * M >> s        Q8 exponent, <= 0 by construction
        e  = 2^(y/256)                    UQ1.15, via `int_exp2_neg`
        p8 = round(e * 255 / 32768)       UINT8

    The exponentials are *not* normalized here. `l = sum(p8)` is returned alongside so the caller
    divides once per row at the end -- which is not a shortcut but a requirement: in the blocked
    dataflow `l` is not final until the row is complete, so normalizing early is not available even
    if it were desirable. Deferring it also makes the factor of 255 cancel between numerator and
    denominator for free (D-006).

    Returns:
        (p8, l) with `p8` UINT8-valued (N, N) and `l` the (N, 1) row sums.
    """
    scores_int = np.asarray(scores_int, dtype=np.int64)

    row_max = scores_int.max(axis=-1, keepdims=True)
    y = apply_multiplier(scores_int - row_max, score_multiplier, score_shift)

    e = int_exp2_neg(y)
    p8 = round_shift(e * UINT8_MAX, EXP_TABLE_BITS)

    # The row max always survives requantization: y = 0 there, so e = 32768 and p8 = 255. A row
    # sum can therefore never be zero, and the final division never divides by zero. This is a
    # property of the construction, not an assumption about the data.
    row_sum = p8.sum(axis=-1, keepdims=True)
    if np.any(row_sum <= 0):
        raise AssertionError("a softmax row summed to zero; the row-max invariant is broken")

    return p8, row_sum
