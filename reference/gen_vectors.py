#!/usr/bin/env python3
"""Generates the golden vectors the C++ simulator is validated against.

A vector set is a directory under `vectors/` holding the *inputs* the accelerator receives, the
*expected output* it has to produce, and the *constants* its datapath needs. All of it comes from
one run of `attention_int8`, so a set is internally consistent by construction: there is no way
for the expected output to drift away from the inputs it was computed from.

The source of truth is FP32. `make_inputs` draws N(0,1) tensors, `attention_int8` quantizes them
and records the scales it chose -- which is the same path R-0 in `docs/RESULTS.md` was measured on.
Writing pre-quantized INT8 inputs instead would skip `choose_scale` and leave the set with no
honest scale to report.

`meta.json` is written for humans. The C++ model has no JSON parser (D-002 forbids the
dependency), so everything the simulator needs arrives as a `.bin` in the ZAAV format of D-001.

Usage:
    python3 reference/gen_vectors.py --smoke        # N=8,  d=8   (toolchain test)
    python3 reference/gen_vectors.py --set small    # N=128, d=64
    python3 reference/gen_vectors.py --set full     # N=1024, d=64
    python3 reference/gen_vectors.py --all
    python3 reference/gen_vectors.py --seq-len 512 --head-dim 64 --name mine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attention_int8 import Int8Attention, attention_int8
from attention_ref import make_inputs
from quantize import (
    EXP_FRAC_BITS,
    EXP_MAX_SHIFT,
    EXP_TABLE_BITS,
    MULT_BITS,
    OUT_FRAC_BITS,
    derive_score_multiplier,
)
from vector_io import VERSION, write_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
VECTORS_DIR = REPO_ROOT / "vectors"

# Predefined vector sets: name -> (seq_len, head_dim)
PRESETS = {
    "smoke": (8, 8),
    "small": (128, 64),
    "full": (1024, 64),
}


# Layout of params.bin. The C++ model reads this array positionally, so the order is part of
# the contract: a new field is appended, never inserted, and removing one bumps the ZAAV version.
PARAM_FIELDS = (
    "score_multiplier",   # 0 — per set: M from derive_score_multiplier
    "score_shift",        # 1 — per set: the right shift that goes with it
    "out_frac_bits",      # 2 — design constant, D-009
    "mult_bits",          # 3 — design constant, width of M
    "exp_frac_bits",      # 4 — design constant, D-004: exponent fraction / table index width
    "exp_table_bits",     # 5 — design constant, D-004: table entries are UQ1.15
    "exp_max_shift",      # 6 — design constant, D-004: below 2^-16 the result is clamped to 0
)


def datapath_params(result: Int8Attention, head_dim: int) -> dict[str, np.ndarray]:
    """The constants the C++ model needs in order to reproduce this set's arithmetic.

    Returns a mapping of file stem -> array; each entry is written as `<stem>.bin`.

    The `(M, shift)` pair is derived *here*, in float64, and crosses the boundary as integers.
    The alternative -- shipping the three float32 scales and letting C++ call its own
    `derive_multiplier` -- was rejected: ZAAV carries float32 while the scales are computed in
    float64, so C++ would re-derive from a degraded number and disagree with this file on 0.135%
    of scale pairs. `math.frexp`, the order of the four float multiplies, and Python's
    round-half-to-even against C's round-half-away-from-zero are three further ways for the two
    sides to part company. D-007 asks for bit-exactness, and the cheapest way to get it is to
    derive once and transmit the result.

    It is also what the hardware does. The accelerator has no FPU: in Phase 2 the ARM core picks
    the scales and folds them into `(M, shift)`, and the fabric only ever sees integers. Asking
    the simulator to re-derive would model something the silicon cannot do.

    The scales themselves are therefore *not* written. Nothing downstream derives anything from
    them -- dequantization is host work and the F0.10 comparison is integer against integer -- so
    they stay in `meta.json` for human readers. If the simulator ever needs to print real units,
    a float32 `scales.bin` can be added without disturbing this file.

    The design constants ride along even though C++ will hardcode its own copies. They are for
    *validation, not configuration*: the C++ constant is the hardware specification, this array is
    a claim about how the vectors were generated, and a disagreement is an error worth reporting
    by name rather than discovering as 11,000 mismatched output values. `main.cpp` already treats
    the vector shapes this way.
    """
    score_multiplier, score_shift = derive_score_multiplier(
        result.scale_q, result.scale_k, head_dim
    )
    values = {
        "score_multiplier": score_multiplier,
        "score_shift": score_shift,
        "out_frac_bits": OUT_FRAC_BITS,
        "mult_bits": MULT_BITS,
        "exp_frac_bits": EXP_FRAC_BITS,
        "exp_table_bits": EXP_TABLE_BITS,
        "exp_max_shift": EXP_MAX_SHIFT,
    }
    return {"params": np.array([values[name] for name in PARAM_FIELDS], dtype=np.int32)}


def write_vector_set(
    name: str,
    seq_len: int,
    head_dim: int,
    seed: int,
    with_intermediates: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Runs the reference pipeline once and writes the whole set to `vectors/<name>/`."""
    q, k, v = make_inputs(seq_len, head_dim, seed=seed)
    result = attention_int8(q, k, v)

    # The inputs the accelerator actually receives, and the output it has to emit. `out` is INT32
    # rather than INT8 because D-009 keeps OUT_FRAC_BITS of fraction, so the emitted value needs
    # 16 bits.
    tensors: dict[str, np.ndarray] = {
        "q": result.q8.astype(np.int8),
        "k": result.k8.astype(np.int8),
        "v": result.v8.astype(np.int8),
        "out": result.out_int.astype(np.int32),
        "row_sums": result.row_sums.astype(np.int32),
    }

    # Stage outputs, so F0.10 can localize a mismatch to a unit instead of reporting that one of
    # three stages is wrong. They dominate the set's size -- both are N*N -- so they are optional.
    # `probs` is UINT8-valued but stored as INT32: D-001 has no unsigned code, and widening the
    # file is preferable to bumping the format version for a debugging aid.
    if with_intermediates:
        tensors["scores"] = result.scores_int.astype(np.int32)
        tensors["probs"] = result.probs_uint8.astype(np.int32)

    tensors.update(datapath_params(result, head_dim))

    out_dir = VECTORS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, array in tensors.items():
        write_tensor(out_dir / f"{stem}.bin", array)

    metadata = {
        "name": name,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "seed": seed,
        "format": {"magic": "ZAAV", "version": VERSION},
        "golden_output": True,
        "scales": {
            "q": result.scale_q,
            "k": result.scale_k,
            "v": result.scale_v,
            "out": result.out_scale,
        },
        "datapath": {
            "layout": list(PARAM_FIELDS),
            "values": {
                name: int(value)
                for name, value in zip(PARAM_FIELDS, tensors["params"].tolist())
            },
        },
        "files": {
            f"{stem}.bin": {"dtype": str(array.dtype), "shape": list(array.shape)}
            for stem, array in sorted(tensors.items())
        },
        "note": "Human-readable summary. The C++ model reads only the .bin files (D-002).",
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return out_dir, tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="shorthand for --set smoke")
    group.add_argument("--set", dest="preset", choices=sorted(PRESETS), help="predefined set")
    group.add_argument("--all", action="store_true", help="generate every predefined set")
    parser.add_argument("--seq-len", type=int, help="custom sequence length (N)")
    parser.add_argument("--head-dim", type=int, help="custom head dimension (d)")
    parser.add_argument("--name", help="name of the custom set")
    parser.add_argument("--seed", type=int, default=0xC0FFEE, help="seed (default: 0xC0FFEE)")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="skip scores.bin and probs.bin, which are O(N^2)")
    args = parser.parse_args()

    if args.seq_len or args.head_dim:
        if not (args.seq_len and args.head_dim and args.name):
            parser.error("--seq-len, --head-dim and --name are used together")
        targets = [(args.name, args.seq_len, args.head_dim)]
    elif args.all:
        targets = [(name, *dims) for name, dims in PRESETS.items()]
    else:
        preset = "smoke" if args.smoke or not args.preset else args.preset
        targets = [(preset, *PRESETS[preset])]

    for name, seq_len, head_dim in targets:
        out_dir, tensors = write_vector_set(
            name, seq_len, head_dim, args.seed,
            with_intermediates=not args.no_intermediates,
        )
        total = sum(a.nbytes for a in tensors.values())
        rel = out_dir.relative_to(REPO_ROOT)
        print(f"[ok] {name}: N={seq_len} d={head_dim} -> {rel}/ "
              f"({len(tensors)} tensors, {total / 1024:.1f} KiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
