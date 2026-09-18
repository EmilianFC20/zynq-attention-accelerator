# Golden vectors

The binary files in this directory are **not versioned** — they are regenerated:

```bash
python3 reference/gen_vectors.py --smoke     # N=8,   d=8
python3 reference/gen_vectors.py --set small # N=128, d=64
python3 reference/gen_vectors.py --set full  # N=1024, d=64
python3 reference/gen_vectors.py --all       # all three
```

The seed is fixed (`0xC0FFEE`), so two generations produce bit-identical results. That's why they
don't need to be versioned: anyone who clones the repo gets exactly the same vectors.

## The three sets and what each one is for

| Set | N | d | On disk | What for |
|---|---|---|---|---|
| `smoke` | 8 | 8 | ~3 KB | Checking that the toolchain works. Useless for measuring anything |
| `small` | 128 | 64 | ~200 KB | Numerical debugging: small enough to inspect by hand, large enough that softmax matters |
| `full` | 1024 | 64 | ~8.4 MB | The real case. Here `S` doesn't fit in BRAM and tiling becomes mandatory |

**All three exist for a concrete reason.** The classic online-softmax bug — forgetting to rescale
the running sum or the output accumulator when a new max appears — produces an almost-correct
result when the max barely moves, i.e. on short sequences. It passes `smoke`, barely passes
`small`, and clearly fails on `full`. Validating only with the small set gives a false sense that
the dataflow works.

## What a set contains

Every file is a ZAAV tensor (see **Format** below). One run of the reference pipeline produces all
of them, so a set is internally consistent by construction — the expected output cannot drift away
from the inputs it was computed from.

| File | Shape | Dtype | Role |
|---|---|---|---|
| `q.bin`, `k.bin`, `v.bin` | (N, d) | int8 | The quantized inputs the accelerator receives |
| `out.bin` | (N, d) | int32 | **The expected output.** What a correct accelerator must emit |
| `row_sums.bin` | (N, 1) | int32 | `ℓ = Σ p8`, the softmax denominators |
| `params.bin` | (7,) | int32 | The datapath constants — see below |
| `scores.bin` | (N, N) | int32 | Stage 1 output, `q8 · k8ᵀ` |
| `probs.bin` | (N, N) | int32 | Stage 2 output, the UINT8 probabilities |
| `meta.json` | — | — | Human-readable summary. Not read by the simulator |

`out.bin` is int32 rather than int8 because the output normalization keeps 8 fractional bits
(D-009), so the emitted value needs 16 bits.

`probs.bin` holds UINT8-valued data in an int32 file: the format has no unsigned code, and widening
the file is preferable to bumping the format version for what is only a debugging aid.

`scores.bin` and `probs.bin` are the two N×N tensors and dominate a set's size — together they are
8 MB of the `full` set's 8.4 MB. They exist so that a mismatch can be localized to a stage instead
of being reported as "the output is wrong". Pass `--no-intermediates` to skip them, which brings
`full` down to about 450 KB.

## `params.bin`

The constants the C++ model needs in order to reproduce a set's arithmetic. Read positionally, so
the order is part of the contract — a new field is appended, never inserted:

| Index | Field | Source |
|---|---|---|
| 0 | `score_multiplier` | Per set. The `M` that carries an INT32 score into the exponent domain |
| 1 | `score_shift` | Per set. The right shift that goes with it |
| 2 | `out_frac_bits` | Design constant, D-009 |
| 3 | `mult_bits` | Design constant, width of `M` |
| 4 | `exp_frac_bits` | Design constant, D-004 |
| 5 | `exp_table_bits` | Design constant, D-004 |
| 6 | `exp_max_shift` | Design constant, D-004 |

Index 0 and 1 are *derived on the Python side and transmitted as integers* rather than being
recomputed from the quantization scales, and the scales themselves are deliberately not written.
The reasoning is D-010.

Indices 2–6 are for **validation, not configuration**. The C++ constants are the hardware
specification; these values are a claim about how the vectors were generated. A disagreement is an
error to report by name, not a parameter for the model to adopt.

## Format

The ZAAV binary format, specified in `docs/DECISIONS.md` (D-001). Mirror implementations in
`reference/vector_io.py` and `model/include/vector_io.hpp`.
