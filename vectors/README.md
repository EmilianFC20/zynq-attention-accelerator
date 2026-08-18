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

| Set | N | d | What for |
|---|---|---|---|
| `smoke` | 8 | 8 | Checking that the toolchain works. Useless for measuring anything |
| `small` | 128 | 64 | Numerical debugging: small enough to inspect by hand, large enough that softmax matters |
| `full` | 1024 | 64 | The real case. Here `S` doesn't fit in BRAM and tiling becomes mandatory |

**All three exist for a concrete reason.** The classic online-softmax bug — forgetting to rescale
the running sum or the output accumulator when a new max appears — produces an almost-correct
result when the max barely moves, i.e. on short sequences. It passes `smoke`, barely passes
`small`, and clearly fails on `full`. Validating only with the small set gives a false sense that
the dataflow works.

## Format

The ZAAV binary format, specified in `docs/DECISIONS.md` (D-001). Mirror implementations in
`reference/vector_io.py` and `model/include/vector_io.hpp`.

Each set directory contains `q.bin`, `k.bin`, `v.bin`, and a `meta.json` with the generation
parameters. From F0.4 onward they will also include the expected output and the quantization
scales.
