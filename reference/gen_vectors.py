#!/usr/bin/env python3
"""Generates the golden vectors the C++ simulator consumes.

Current status (F0.1): generates the **inputs** (Q, K, V) in INT8, which is all that is needed to
validate the I/O contract between Python and C++.

Pending in F0.4: also dump the expected output and the quantization scales. That depends on F0.2
(FP32 reference) and F0.3 (INT8 scheme), which are not implemented yet.

Usage:
    python3 reference/gen_vectors.py --smoke        # N=8,  d=8   (toolchain test)
    python3 reference/gen_vectors.py --set small    # N=128, d=64
    python3 reference/gen_vectors.py --set full     # N=1024, d=64
    python3 reference/gen_vectors.py --seq-len 512 --head-dim 64 --name mine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vector_io import write_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
VECTORS_DIR = REPO_ROOT / "vectors"

# Predefined vector sets: name -> (seq_len, head_dim)
PRESETS = {
    "smoke": (8, 8),
    "small": (128, 64),
    "full": (1024, 64),
}


def generate_inputs(seq_len: int, head_dim: int, seed: int) -> dict[str, np.ndarray]:
    """Generates Q, K, V in INT8 from a fixed seed.

    A clipped normal is used rather than a uniform because real transformer activations are far
    from uniform, and a realistic input range matters when the quantization scales have to be
    chosen in F0.3.
    """
    rng = np.random.default_rng(seed)
    shape = (seq_len, head_dim)

    def sample() -> np.ndarray:
        raw = rng.normal(loc=0.0, scale=40.0, size=shape)
        return np.clip(np.rint(raw), -127, 127).astype(np.int8)

    return {"q": sample(), "k": sample(), "v": sample()}


def write_vector_set(name: str, seq_len: int, head_dim: int, seed: int) -> Path:
    out_dir = VECTORS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = generate_inputs(seq_len, head_dim, seed)
    for tensor_name, array in tensors.items():
        write_tensor(out_dir / f"{tensor_name}.bin", array)

    metadata = {
        "name": name,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "seed": seed,
        "dtype": "int8",
        "tensors": sorted(tensors),
        "golden_output": False,  # becomes True in F0.4
        "note": "Inputs only. The expected output is added in F0.4 (requires F0.2 and F0.3).",
    }
    (out_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return out_dir


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
        out_dir = write_vector_set(name, seq_len, head_dim, args.seed)
        rel = out_dir.relative_to(REPO_ROOT)
        print(f"[ok] {name}: N={seq_len} d={head_dim} -> {rel}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
