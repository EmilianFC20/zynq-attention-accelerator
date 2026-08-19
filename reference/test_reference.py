#!/usr/bin/env python3
"""Self-validation for the FP32 attention reference.

Run it directly, no test framework required (same philosophy as the C++ side, see
docs/DECISIONS.md D-002):

    python3 reference/test_reference.py

Exits non-zero on the first failing check so it can be wired into CI later.
"""

from __future__ import annotations

import math

import torch

from attention_ref import attention_fp32, make_inputs, stable_softmax, validate_against_torch

# Tolerance for "our FP32 result equals PyTorch's FP32 result". The two differ only by
# accumulation order inside the matmuls, so the gap grows slowly with N (more terms summed) but
# stays several orders of magnitude below the values themselves.
FP32_TOL = 1e-5

_failures = 0


def check(condition: bool, description: str) -> None:
    global _failures
    if condition:
        print(f"  [ok]   {description}")
    else:
        _failures += 1
        print(f"  [FAIL] {description}")


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def test_shapes() -> None:
    print("shapes and dtypes")
    n, d = 16, 8
    q, k, v = make_inputs(n, d)
    out, scores, probs = attention_fp32(q, k, v)

    check(tuple(out.shape) == (n, d), f"out is (N, d) = ({n}, {d})")
    check(tuple(scores.shape) == (n, n), f"scores is (N, N) = ({n}, {n})")
    check(tuple(probs.shape) == (n, n), f"probs is (N, N) = ({n}, {n})")
    check(out.dtype == torch.float32, "out stays float32")


def test_scaling() -> None:
    """The 1/sqrt(d) factor is in the right place, computed independently of torch matmul."""
    print("score scaling")
    n, d = 8, 8
    q, k, v = make_inputs(n, d)
    _, scores, _ = attention_fp32(q, k, v)

    # Same quantity via an explicit loop: no matmul, no broadcasting, nothing shared with the
    # implementation under test.
    manual = sum(float(q[0, t]) * float(k[3, t]) for t in range(d)) / math.sqrt(d)
    check(close(float(scores[0, 3]), manual, 1e-5), "scores[0,3] == dot(q[0], k[3]) / sqrt(d)")


def test_softmax_is_a_distribution() -> None:
    print("softmax produces valid distributions")
    n, d = 32, 16
    q, k, v = make_inputs(n, d)
    _, _, probs = attention_fp32(q, k, v)

    row_sums = probs.sum(dim=-1)
    check(bool(torch.allclose(row_sums, torch.ones(n), atol=1e-6)), "every row sums to 1")
    check(bool((probs >= 0).all()), "no negative probabilities")
    check(bool(torch.isfinite(probs).all()), "no NaN or inf")


def test_softmax_stability() -> None:
    """The whole point of subtracting the row max: naive exp() would overflow here."""
    print("softmax numerical stability")
    scores = torch.tensor([[1000.0, 999.0, 998.0],
                           [-1000.0, -1001.0, -1002.0]], dtype=torch.float32)
    probs = stable_softmax(scores)

    check(bool(torch.isfinite(probs).all()), "no NaN with |scores| ~ 1000 (naive exp overflows)")
    check(bool(torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-6)), "rows still sum to 1")
    check(bool((probs[0, 0] > probs[0, 1]) and (probs[0, 1] > probs[0, 2])), "ordering preserved")

    # A constant added to a whole row must not change the result — that invariant is exactly what
    # makes the max-subtraction legal, and what the online softmax relies on when it rescales.
    shifted = stable_softmax(scores + 500.0)
    check(bool(torch.allclose(probs, shifted, atol=1e-6)), "shift-invariant across the row")


def test_determinism() -> None:
    print("determinism")
    q1, k1, v1 = make_inputs(16, 8, seed=123)
    q2, k2, v2 = make_inputs(16, 8, seed=123)
    check(bool(torch.equal(q1, q2) and torch.equal(k1, k2) and torch.equal(v1, v2)),
          "make_inputs is bit-identical for the same seed")

    out1, _, _ = attention_fp32(q1, k1, v1)
    out2, _, _ = attention_fp32(q2, k2, v2)
    check(bool(torch.equal(out1, out2)), "attention_fp32 is bit-identical across runs")


def test_matches_pytorch() -> None:
    print(f"agreement with torch SDPA (tolerance {FP32_TOL:g})")
    for n, d in [(8, 8), (128, 64), (256, 64)]:
        q, k, v = make_inputs(n, d)
        diff = validate_against_torch(q, k, v)
        check(diff <= FP32_TOL, f"N={n:<4} d={d:<3} max|diff| = {diff:.3e}")


def main() -> int:
    for test in (
        test_shapes,
        test_scaling,
        test_softmax_is_a_distribution,
        test_softmax_stability,
        test_determinism,
        test_matches_pytorch,
    ):
        test()
        print()

    if _failures:
        print(f"{_failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
