#!/usr/bin/env python3
"""Self-validation for the INT8 quantization scheme and the integer attention pipeline.

Run it directly, no test framework required (docs/DECISIONS.md D-002):

    python3 reference/test_quantize.py

Exits non-zero on the first failing check so it can be wired into CI later.

What this file checks is *integer arithmetic behaving as specified*, not accuracy. The accuracy of
the INT8 pipeline against FP32 is a measurement reported in docs/RESULTS.md, deliberately kept
separate from pass/fail (D-007) -- a design can be perfectly implemented and still lose 2% to
quantization, and conflating the two makes both claims unfalsifiable.
"""

from __future__ import annotations

import math

import numpy as np

from attention_int8 import attention_int8, quantization_error
from attention_ref import make_inputs
from quantize import (
    EXP2_LUT,
    EXP_FRAC_BITS,
    EXP_MAX_SHIFT,
    EXP_TABLE_BITS,
    INT8_MAX,
    MULT_BITS,
    OUT_FRAC_BITS,
    UINT8_MAX,
    apply_multiplier,
    choose_scale,
    derive_multiplier,
    derive_score_multiplier,
    dequantize,
    floor_div,
    int_exp2_neg,
    int_softmax,
    quantize_symmetric,
    round_shift,
)

_failures = 0


def check(condition: bool, description: str) -> None:
    global _failures
    if condition:
        print(f"  [ok]   {description}")
    else:
        _failures += 1
        print(f"  [FAIL] {description}")


def test_round_shift() -> None:
    """D-005: round-half-up, arithmetic, identical in Python, C++20 and Verilog."""
    print("round_shift (D-005)")

    check(round_shift(8, 1) == 4, "8 >> 1 = 4")
    check(round_shift(9, 1) == 5, "9 >> 1 rounds up to 5, not down to 4")
    check(round_shift(-9, 1) == -4, "-9 >> 1 = -4 (half-up, so toward +inf on a tie)")
    check(round_shift(-8, 1) == -4, "-8 >> 1 = -4 exactly")

    # The bias argument for the rule: over a uniform sweep, truncation loses half an LSB every
    # time and the errors add up; rounding cancels everywhere except on exact ties.
    values = np.arange(-2048, 2048)
    rounded_bias = float((round_shift(values, 4) - values / 16).sum())
    truncated_bias = float(((values >> 4) - values / 16).sum())
    check(abs(rounded_bias) < abs(truncated_bias) / 10,
          f"rounding bias {rounded_bias:+.0f} vs truncation bias {truncated_bias:+.0f}")

    # The residue is not noise, it is exactly D-005's acknowledged cost: half-up breaks ties in
    # one direction, so each of the 4096/16 exact ties contributes +1/2. Round-half-to-even would
    # zero this out and cost tie-detection logic in three languages -- the trade D-005 declined.
    ties = len(values) // 16
    check(rounded_bias == ties * 0.5,
          f"the whole residual bias is the {ties} exact ties, at +1/2 each")

    try:
        round_shift(4, 0)
        check(False, "s = 0 is rejected")
    except ValueError:
        check(True, "s = 0 is rejected (no half-LSB exists at s = 0)")


def test_floor_div() -> None:
    """The one division the shift rule cannot reach; C++ must match this, not its own `/`."""
    print("floor_div")
    check(floor_div(7, 2) == 3, "7 // 2 = 3")
    check(floor_div(-7, 2) == -4, "-7 // 2 = -4 (floor), where C++ '/' would give -3")
    check(floor_div(np.array([-7, -1, 0, 1, 7]), 2).tolist() == [-4, -1, 0, 0, 3],
          "vectorized floor semantics across the sign change")


def test_choose_scale() -> None:
    """D-008: absolute-maximum calibration, so nothing clips."""
    print("choose_scale (D-008)")

    x = np.array([-3.0, 0.5, 2.0])
    check(math.isclose(choose_scale(x), 3.0 / INT8_MAX), "scale = absmax / 127")

    x_int, scale = quantize_symmetric(x)
    check(int(np.abs(x_int).max()) == INT8_MAX, "the extreme element lands exactly on 127")
    check(bool((np.abs(x_int) <= INT8_MAX).all()), "nothing exceeds the symmetric range")

    zeros = np.zeros((4, 4))
    zero_scale = choose_scale(zeros)
    z_int, _ = quantize_symmetric(zeros)
    check(zero_scale > 0.0, "an all-zero tensor still gets a positive scale")
    check(bool(np.isfinite(z_int).all()) and int(np.abs(z_int).max()) == 0,
          "an all-zero tensor quantizes to zeros, not NaN")

    # Round-trip error is bounded by half a step, by construction -- that bound is the entire
    # accuracy claim of abs-max calibration.
    q, _, _ = make_inputs(64, 32)
    q = q.numpy().astype(np.float64)
    q_int, q_scale = quantize_symmetric(q)
    check(float(np.abs(dequantize(q_int, q_scale) - q).max()) <= q_scale / 2 + 1e-12,
          "round-trip error never exceeds half a quantization step")


def test_derive_multiplier() -> None:
    """The fixed-point constant idiom: scale ~= M >> s, with M narrow enough for a DSP48E1."""
    print("derive_multiplier")

    for scale in (0.5, 0.1, 1e-4, 0.999999, 1.0, 3.7e-6):
        m, s = derive_multiplier(scale)
        check((1 << (MULT_BITS - 1)) <= m < (1 << MULT_BITS),
              f"scale {scale:g}: M = {m} is exactly {MULT_BITS} bits wide")
        check(abs(m / (1 << s) - scale) <= scale * 2.0 ** -(MULT_BITS - 1),
              f"scale {scale:g}: M >> {s} reconstructs it to within 2^-{MULT_BITS - 1}")

    # The renormalization branch: a mantissa that rounds up to exactly 1.0.
    m, s = derive_multiplier(2.0 ** -3 * (1 - 2.0 ** -20))
    check(m < (1 << MULT_BITS), "a mantissa rounding to 1.0 renormalizes instead of overflowing")

    for bad in (0.0, -1.0, float("inf"), float("nan")):
        try:
            derive_multiplier(bad)
            check(False, f"{bad} is rejected")
        except ValueError:
            check(True, f"scale {bad} is rejected before it can poison the datapath")

    check(apply_multiplier(1000, *derive_multiplier(0.25)) == 250,
          "apply_multiplier(1000, 0.25) = 250")


def test_exp2_table() -> None:
    """D-004: 256 entries of 2^(-f/256) in UQ1.15."""
    print("exp2 table (D-004)")

    check(len(EXP2_LUT) == (1 << EXP_FRAC_BITS), f"{1 << EXP_FRAC_BITS} entries")
    check(EXP2_LUT[0] == (1 << EXP_TABLE_BITS), "entry 0 is exactly 1.0 (32768)")
    check(int(EXP2_LUT.max()) == (1 << EXP_TABLE_BITS) and int(EXP2_LUT.min()) > (1 << 14),
          "every entry lies in (16384, 32768], so 16 bits suffice")
    check(bool((np.diff(EXP2_LUT) <= 0).all()), "monotonically decreasing")

    exact = np.exp2(-np.arange(1 << EXP_FRAC_BITS) / (1 << EXP_FRAC_BITS))
    check(float(np.abs(EXP2_LUT / (1 << EXP_TABLE_BITS) - exact).max()) <= 2.0 ** -(EXP_TABLE_BITS + 1),
          "every entry within half an LSB of the true value")

    # 512 bytes: the argument that it lives in LUTRAM and costs no BRAM.
    check(len(EXP2_LUT) * 2 == 512, "the table is 512 bytes")


def test_int_exp2_neg() -> None:
    """The shift-plus-lookup decomposition, against float exp2 as the oracle."""
    print("int_exp2_neg")

    y = -np.arange(0, 4096)
    got = int_exp2_neg(y).astype(np.float64) / (1 << EXP_TABLE_BITS)
    want = np.exp2(y / (1 << EXP_FRAC_BITS))

    covered = np.arange(len(y)) < EXP_MAX_SHIFT * (1 << EXP_FRAC_BITS)
    check(float(np.abs(got[covered] - want[covered]).max()) < 2e-4,
          "matches float 2^(y/256) across the whole represented range")
    check(int(int_exp2_neg(np.array([0]))[0]) == (1 << EXP_TABLE_BITS), "2^0 = 1.0 exactly")
    check(bool((int_exp2_neg(y[~covered]) == 0).all()),
          f"clamped to 0 past n >= {EXP_MAX_SHIFT}, rather than shifting into C++ UB")

    try:
        int_exp2_neg(np.array([1]))
        check(False, "a positive exponent is rejected")
    except ValueError:
        check(True, "a positive exponent is rejected (the row max must be subtracted first)")


def test_int_softmax_invariants() -> None:
    """Structural properties of the integer softmax, checked without reference to FP32.

    These are the checks that still work in F1.7, when the thing under test is Verilog and there
    is no float oracle to compare against. Mutation-tested: reducing the row max along the wrong
    axis, reversing the exp2 table, and corrupting a single table entry are each caught by two of
    them independently.

    What they deliberately do *not* catch is worth stating, because a test whose blind spots are
    unknown is worse than one whose blind spots are documented. A table that is wrong but still
    well-shaped -- every entry flattened to 1.0, say -- satisfies every invariant here, and so does
    truncating instead of rounding when requantizing to UINT8. Neither is a gap in coverage
    overall: the first is caught by `test_exp2_table` and `test_int_exp2_neg`, which compare the
    table against its closed form entry by entry, and the second by `test_round_shift`. The
    division of labour is deliberate -- value checks verify the *arithmetic*, these verify the
    *structure*, and the RTL will only have the second kind available to it.
    """
    print("int_softmax invariants")

    q, k, _ = make_inputs(64, 32)
    q8, sq = quantize_symmetric(q.numpy().astype(np.float64))
    k8, sk = quantize_symmetric(k.numpy().astype(np.float64))
    scores = q8 @ k8.T
    p8, row_sums = int_softmax(scores, *derive_score_multiplier(sq, sk, 32))

    # 1. Range. UINT8 is the format (D-006), so this is the floor of any correctness claim: a
    #    value outside it means the requantization shift is wrong and the C++ model would be
    #    writing into the neighbouring byte.
    check(int(p8.min()) >= 0 and int(p8.max()) <= UINT8_MAX,
          f"every probability lies in [0, {UINT8_MAX}]")

    # 2. The row-max invariant. y = 0 at the largest score, so e = 32768 and p8 = 255 exactly.
    #    Everything downstream leans on this: it is what guarantees no row can sum to zero, and
    #    therefore that the final division never divides by zero, without assuming anything about
    #    the data. If the row-max subtraction is dropped or the wrong axis is reduced, this is the
    #    check that goes red first.
    check(bool((p8.max(axis=-1) == UINT8_MAX).all()),
          f"the largest score in every row requantizes to exactly {UINT8_MAX}")

    # 3. Monotonicity. A larger score must never produce a smaller probability. This is the check
    #    that catches a sign error, an inverted table, or a shift applied in the wrong direction --
    #    all of which can leave the range and the row max intact while scrambling the ordering.
    #    Non-strict, because rounding legitimately maps neighbouring scores onto the same code.
    ordered = np.take_along_axis(p8, np.argsort(scores, axis=-1), axis=-1)
    check(bool((np.diff(ordered, axis=-1) >= 0).all()),
          "sorting a row by score gives non-decreasing probabilities")

    # 4. Shift invariance. Adding a constant to a whole row must not change p8 by a single bit --
    #    the row max moves with it and the difference is untouched. This is the algebraic property
    #    that makes the max-subtraction legal in the first place, and the same one the online
    #    softmax of F0.9 will rely on when it rescales partial accumulators against a new running
    #    max. Testing it here means F0.9 inherits a checked assumption rather than a hoped-for one.
    shifted, _ = int_softmax(scores + 5000, *derive_score_multiplier(sq, sk, 32))
    check(bool(np.array_equal(p8, shifted)),
          "adding a constant to a row leaves p8 bit-identical")

    # 5. row_sums is what it claims to be. Cheap, and it is the value the whole output is divided
    #    by -- a stale or mis-reduced denominator would scale an entire row of the output without
    #    breaking any other property here.
    check(bool(np.array_equal(row_sums, p8.sum(axis=-1, keepdims=True))),
          "row_sums is the row-wise sum of p8, reduced along the right axis")

    # 6. The consequence of (2) and (5) stated directly, because it is the one a reader of the
    #    final division actually needs: the denominator is bounded away from zero by construction.
    check(int(row_sums.min()) >= UINT8_MAX,
          f"no row sums below {UINT8_MAX}, so the output division can never divide by zero")


def test_pipeline_shapes_and_ranges() -> None:
    """The INT8 pipeline emits what the format promises, at every stage."""
    print("INT8 pipeline stages")

    n, d = 32, 16
    q, k, v = make_inputs(n, d)
    r = attention_int8(q, k, v)

    check(r.scores_int.shape == (n, n), f"scores_int is (N, N) = ({n}, {n})")
    check(r.probs_uint8.shape == (n, n), f"probs_uint8 is (N, N) = ({n}, {n})")
    check(r.out_int.shape == (n, d), f"out_int is (N, d) = ({n}, {d})")
    check(r.scores_int.dtype.kind == "i" and r.probs_uint8.dtype.kind == "i",
          "every intermediate is an integer type, not a float")

    check(int(np.abs(r.scores_int).max()) < 2 ** 31,
          "the score accumulator stays inside INT32 (headroom is the point, D-006)")
    check(0 <= int(r.probs_uint8.min()) and int(r.probs_uint8.max()) <= UINT8_MAX,
          "probabilities occupy [0, 255]")
    check(math.isclose(r.out_scale, r.scale_v / (1 << OUT_FRAC_BITS)),
          f"the output LSB is s_v / {1 << OUT_FRAC_BITS} (D-009)")


def test_determinism() -> None:
    print("determinism")
    q, k, v = make_inputs(32, 16, seed=99)
    a, b = attention_int8(q, k, v), attention_int8(q, k, v)
    check(bool(np.array_equal(a.out_int, b.out_int)),
          "identical inputs give bit-identical integer outputs")
    check(bool(np.array_equal(a.probs_uint8, b.probs_uint8)),
          "and bit-identical probabilities (nothing here depends on float ordering)")


def test_accuracy_is_reported_not_asserted() -> None:
    """Measurement, not a gate (D-007). The check is only that it stays in a sane band."""
    print("accuracy against FP32 (measured, see docs/RESULTS.md)")
    for n, d in [(128, 64), (256, 64)]:
        error = quantization_error(n, d)
        print(f"         {error}")
        check(error.relative_rms < 0.05,
              f"N={n}: relative RMS {error.relative_rms:.4f} is within the reported band")


def main() -> int:
    for test in (
        test_round_shift,
        test_floor_div,
        test_choose_scale,
        test_derive_multiplier,
        test_exp2_table,
        test_int_exp2_neg,
        test_int_softmax_invariants,
        test_pipeline_shapes_and_ranges,
        test_determinism,
        test_accuracy_is_reported_not_asserted,
    ):
        try:
            test()
        except NotImplementedError as pending:
            global _failures
            _failures += 1
            print(f"  [FAIL] pending: {pending}")
        print()

    if _failures:
        print(f"{_failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
