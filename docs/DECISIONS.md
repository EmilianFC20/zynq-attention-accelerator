# Design decisions

An ADR-style log: every non-obvious decision, with its reasoning and the alternatives rejected.

This file exists to document all the decisions that were made when creating the project.

---

## D-001 — Custom binary format for the golden vectors

**Date:** 2026-08-02 · **Status:** accepted

**Decision.** A minimal, custom binary format for passing tensors from Python to C++:

```
offset  size    field
0       4       magic "ZAAV" (ASCII)
4       4       version (uint32 little-endian, currently 1)
8       4       dtype   (uint32: 0=int8, 1=int32, 2=float32)
12      4       ndim    (uint32, max 4)
16      16      dims[4] (uint32 ×4, unused dimensions are 1)
32      ...     raw data, C-contiguous, little-endian
```

**Why.** The C++ model can't have external dependencies (see D-002), which rules out reading `.npy` with a library or using Protobuf/HDF5. Writing a `.npy` parser by hand is possible, but its header is a Python dict literal — you'd have to parse text in order to read a binary file, which is absurd. Thirty-two bytes of fixed-width header are read with one `fread` and validated with two `if`s.

**Alternatives rejected.** `.npy` (text header, needlessly fragile parser); CSV (loses floating-point precision, enormous files at N=4096); HDF5 (heavy dependency).

**Consequence.** Any change to the format has to be made on both sides at once, and the `version` field exists to catch the mismatch instead of silently reading garbage.

---

## D-002 — The C++ model has no external dependencies

**Date:** 2026-08-02 · **Status:** accepted

**Decision.** `model/` builds with nothing but `g++ -std=c++17` and a flat Makefile. No cmake, no
third-party library, no package manager.

**Why.** Three reasons, in order of importance:
1. **It's a portfolio piece.** Someone evaluating the repo should be able to clone it and run
   `make` without fighting dependencies. Every install step loses a fraction of your readers, and
   the readers you lose are exactly the ones who were going to look at it for five minutes
   between meetings.
2. **Portability into the Phase 1 flow.** The same code will be linked against the Verilator harness for co-simulation. Fewer dependencies, less friction at that point.
3. `cmake` isn't installed on the development machine, and a 30-line Makefile more than covers a project this size.

**Consequence.** No `fmt`, no `Eigen`, no test framework. The test harness is a ten-line `check()`
function. That's enough.

---

## D-003 — 12×12 systolic array as the baseline, 16×16 as a stretch goal

**Date:** 2026-08-02 · **Status:** accepted (to be revisited in F1.3)

**Decision.** The Phase 1 RTL targets a **12×12** array (144 MACs). The **16×16** array
(256 MACs) is a stretch goal, conditional on INT8 packing.

**Why.** The XC7Z020 has **220 DSP48E1 slices**. A 16×16 array needs 256 MACs, which at one MAC
per DSP **does not fit**. Xilinx documents a packing technique that exploits the DSP48E1's 25×18
bit multiplier to do two INT8 multiplications in a single slice, which would bring 16×16 down to
~128 DSPs — comfortable. But that technique complicates the datapath, and it isn't worth risking
the schedule for it before the full path is working. 12×12 = 144 DSPs fits directly, leaves 76
DSPs of headroom for softmax and control logic, and is a perfectly defensible size.

**Consequence.** The Phase 0 model must sweep both sizes so the final decision is made from data
rather than from a hunch.

---

## D-004 — Integer-only softmax with a 256-entry `exp2` lookup table

**Date:** 2026-08-19 · **Status:** accepted; table size confirmed at 256 entries by the F0.3
measurement (see D-008)

**Decision.** Softmax is computed entirely in integer arithmetic. The exponential is evaluated as
a base-2 power, split into a shift and a small table lookup:

```
exp(x) = 2^(x · log₂e),    log₂e ≈ 1.442695   (folded into the score scale, free)

y = (S_int32 − rowmax) · M >> s      # exponent × log₂e, 8 fractional bits, always ≤ 0
n = (−y) >> 8                        # integer part → right shift
f = (−y) & 0xFF                      # fractional part → table index
e = EXP2_LUT[f] >> n                 # result in UQ1.15;  n ≥ 16 → e = 0
```

`EXP2_LUT[f] = round(2^(−f/256) · 32768)` for `f = 0…255`: **256 entries × 16 bits = 512 bytes**.

**Why.** The exponential is not linear, so unlike the two matmuls it cannot be handled by
factoring a scale out of an exact integer accumulation. It has to be approximated somewhere, and
the target device decides where: the PL half of the XC7Z020 has **no floating-point hardware**.

The table stays small because it only has to cover a domain of width 1. After the row-max
subtraction, `2^(−f)` for `f ∈ [0,1)` lies in `(0.5, 1]` — the entire remaining dynamic range is
`2^(−n)`, which is a barrel shift and costs nothing. The same max-subtraction that makes softmax
numerically stable is what makes it cheap in hardware.

Sizing rationale, and why 256 entries rather than more: an 8-bit index bounds the interpolation
error at `ln2 / 512 ≈ 1.35e-3` absolute, which against table values in `(0.5, 1]` is a relative
error of at most **≈2.7e-3**. The UINT8 format that probabilities are requantized into (D-006)
has a step of `1/255 ≈ 3.9e-3`. The table error therefore sits just under the noise floor the
output format already imposes, so additional entries buy nothing until `P` gets wider. At 512
bytes the table lives in distributed LUTRAM and does not consume any of the 140 BRAM blocks.

UQ1.15 rather than UQ0.16 because `1.0` in UQ0.16 is 65536, which does not fit in 16 bits and
would force a saturation special case. In UQ1.15 the table range is `(16384, 32768]` — clean.

**Alternatives rejected.**

- **Requantize and do softmax in floating point.** On this device that means either building FP
  units in fabric, or shipping `S` back to the PS over AXI. The second is self-defeating: at
  N=1024 it moves ~2 MB of score matrix across the bus, which is precisely the traffic the
  blocked dataflow exists to eliminate. Avoiding a 512-byte table by spending 2 MB of bandwidth is
  the wrong trade by four orders of magnitude.
- **A direct `exp()` table over [−12, 0]** at the same fractional resolution needs ~3072 entries
  and pins a BRAM, for no accuracy gain over the shift-plus-table decomposition.
- **Polynomial approximation** (as in integer-only BERT work) is more accurate per bit stored, but
  needs multipliers and a term-by-term fixed-point error analysis. Worse effort-to-result ratio at
  this precision target.

**Consequence.** The exponent path needs an integer multiplier `M` and shift `s` derived from the
score scale — the standard fixed-point requantization idiom, and the same one D-005 governs. The
clamp at `n ≥ 16` must be explicit rather than left to the shift: `2^(−16) ≈ 1.5e-5` is already
below UINT8 resolution, and in C++ a shift of a 32-bit value by ≥ 32 is undefined behavior, which
would surface as an unreproducible model-versus-RTL mismatch. If the F0.3 error measurement comes
out worse than useful, the first lever is 256 → 512 entries (1 KB, still no BRAM), and the
resulting accuracy-versus-storage curve becomes a reportable result rather than a guess.

---

## D-005 — One rounding rule for the entire datapath: round-half-up

**Date:** 2026-08-19 · **Status:** accepted

**Decision.** Every fixed-point right-shift in the design rounds by adding a half-LSB and then
shifting **arithmetically**:

```
round_shift(x, s) = (x + (1 << (s−1))) >> s
```

This applies identically in the Python reference, the C++ model, and the Verilog. No site is
allowed to truncate, and no site is allowed to round differently.

**Why.** Truncation loses on average half an LSB *in the same direction every time*. That bias
accumulates linearly with the number of terms, while random rounding error grows only as `√N` —
at N=1024 the difference is not academic. Rounding costs one adder and one shift.

The rule is chosen as much for reproducibility as for accuracy. It expresses identically in all
three languages, which is what makes the bit-exact comparison of D-007 achievable:

| | expression | note |
|---|---|---|
| Python | `(x + (1 << (s-1))) >> s` | `>>` on negative ints floors — already arithmetic |
| C++ | `(x + (1 << (s-1))) >> s` | arithmetic shift on signed types is mandated in C++20 |
| Verilog | `(x + (1 <<< (s-1))) >>> s` | `>>>` with a `signed` operand |

**Alternatives rejected.** **Round-half-to-even** (banker's rounding, what IEEE 754 does) is unbiased even on exact ties, but requires tie detection plus an LSB inspection: more logic, and three more opportunities for the implementations to differ subtly. Ties are rare in this datapath.
The gemmlowp/TFLite requantization path makes the same trade. **Truncation** is rejected on the bias argument above.

**Consequence.** The one place this rule does not reach is the final normalization `O = O_acc / ℓ`, which is a true division rather than a shift, and whose numerator is signed because `V` is signed. C++ integer division truncates toward zero while Python's `//` floors toward −∞ — they disagree on negative operands. The division must therefore be specified as an explicit formula and implemented through a shared floor-division helper on both sides. This is the most likely single source of a Phase 1 co-simulation mismatch, and it is cheap to prevent now and expensive to find later.

---

## D-006 — Probabilities are UINT8; the PE multiplier is 9-bit × 8-bit signed

**Date:** 2026-08-19 · **Status:** accepted (operand widths to be confirmed in F1.3)

**Decision.** The *unnormalized* exponentials are requantized to unsigned 8-bit,
`p8 = round(e · 255)` with `e ∈ (0,1]`, and fed directly into the second matmul. Normalization
happens once per row at the end, dividing by `ℓ = Σ p8`. The processing element is built with a
**9-bit signed × 8-bit signed** multiplier so that one systolic array serves both GEMMs.

**Why.** Three things fall out of this at once.

*The 255 factor cancels.* The output is `O = (Σ p8ⱼ · vⱼ) / (Σ p8ⱼ)`. Numerator and denominator
carry the same scale, so it vanishes with no correction term and no extra multiplier. Quantizing
before normalizing is free.

*Deferring the division is what the blocked dataflow requires anyway.* `ℓ` is not final until the
row completes, so probabilities cannot be normalized early. What looks like a hardware compromise
is the algorithm's own structure: one reciprocal per row and `d` multiplies, instead of `N²`
divisions.

*Unsigned buys a full extra bit for free.* A probability is known non-negative and known bounded
by 1 — unlike Q, K and V, its range needs no calibration. Spending a sign bit on a quantity that
is never negative is waste.

The consequence is that the second matmul is unsigned × signed while the first is signed ×
signed. A 9-bit signed operand covers both cases: `{1'b0, p8}` for probabilities, sign-extended
INT8 for queries. The DSP48E1 multiplier is 25 × 18 (D-003), so the ninth bit costs nothing in
DSP terms, and one PE design and one array handle both matmuls.

**Alternatives rejected.**

- **UQ0.7 signed probabilities**, so a plain symmetric INT8 × INT8 PE suffices. Rejected: it
  nearly doubles the probability quantization step (`1/127 ≈ 7.9e-3` against `1/255 ≈ 3.9e-3`),
  pushing it above the LUT error of D-004 and wasting that table's precision — in exchange for
  avoiding a change that is free on a 25×18 multiplier.
- **Keeping the exponentials at UQ1.15 into the matmul.** More accurate, and a 16 × 8 product
  still fits one DSP, but the second matmul would then need a different PE than the first,
  forfeiting array reuse. On a 220-DSP budget that is a bad trade, and it complicates F1.3 and
  F1.6 substantially.

**Consequence.** Attention's two GEMMs have genuinely different operand statistics, and the design
acknowledges it rather than forcing symmetry. The PE must be parameterized with independent
operand widths from the start, and the C++ model has to track signedness per operand rather than
assuming `int8_t` throughout.

---

## D-007 — Bit-exact verification; accuracy measured separately

**Date:** 2026-08-19 · **Status:** accepted

**Decision.** Two different questions are asked with two different instruments, and they are never
conflated:

| Question | Comparison | Criterion |
|---|---|---|
| **Correctness** | C++ model vs Python reference; RTL vs C++ model | **Bit-exact. Zero tolerance.** |
| **Accuracy** | INT8 pipeline vs the FP32 reference of F0.2 | Measured, reported, never a gate |

The Python reference implements the integer pipeline of D-004 through D-006 exactly — the same
table, the same rounding, the same widths — rather than computing softmax in floating point. F0.10
and F1.7 assert equality, not proximity.

**Why.** If the reference used floating-point softmax, every downstream comparison would have to
carry a tolerance wide enough to absorb the quantization difference. A genuine bug that perturbs
an output by less than that band would then be invisible, and no amount of testing could
distinguish "the model is correct" from "the model is wrong by less than the noise." The claim
that the simulator matches the reference would become unfalsifiable, and an unfalsifiable claim is
not a verification result.

Modelling the integer pipeline exactly inverts this: any mismatch, down to one bit, is a bug and
is locatable. That is only possible because every approximation in the datapath is *deterministic*
— a table lookup and a shift, not a library kernel whose accumulation order may vary.

The accuracy question is real and still gets answered, just separately: how far the quantized
pipeline lands from FP32 is a **property of the design** to be measured and reported in
`RESULTS.md`, not a threshold anything passes or fails.

**Alternatives rejected.** **A single tolerance covering both questions** — simpler to implement,
and the reason it is rejected is the whole argument above. **Floating-point softmax in the
reference with a tight tolerance downstream** merely hides the same problem behind a smaller
number.

**Consequence.** The reference gets slower and more intricate: integer-exact NumPy instead of one
call to `torch.softmax`. Any later change to the table, the rounding, or the operand widths
invalidates the golden vectors, which must be regenerated and re-verified — the `version` field of
D-001 exists to catch a stale set instead of silently comparing against the wrong truth. In
exchange, "the C++ model reproduces the reference bit for bit, and the RTL reproduces the model
bit for bit" becomes a statement that can actually be defended.

---

## D-008 — Absolute-maximum calibration for the Q, K, V scales

**Date:** 2026-09-06 · **Status:** accepted

**Decision.** The per-tensor scale is `absmax(x) / 127`. Nothing clips, and the only error
`quantize_symmetric` introduces is the half-LSB of the rounding itself. An all-zero tensor falls
back to `1/127` so the scale stays positive and the downstream division stays finite.

**Why.** The alternative worth taking seriously is percentile clipping — calibrate on, say, the
99.9th percentile of `|x|`, deliberately saturating the tail to buy a finer grid for the bulk of
the distribution. Measured on the project's own inputs, it loses on every metric:

| N, d | calibration | SQNR (dB) | values clipped | max abs error on `S` | relative RMS on `S` |
|---|---|---|---|---|---|
| 128, 64 | **abs-max** | **41.03** | 0 | **0.063** | **0.0129** |
| 128, 64 | p99.9 | 37.72 | 9 / 8192 | 0.286 | 0.0176 |
| 128, 64 | p99 | 27.98 | 82 / 8192 | 0.850 | 0.0703 |
| 1024, 64 | **abs-max** | **39.34** | 0 | — | — |
| 1024, 64 | p99.9 | 36.48 | 66 / 65536 | — | — |
| 1024, 64 | p99 | 27.04 | 655 / 65536 | — | — |

Percentile clipping is a trade that only pays when a distribution has a long, thin tail wasting
the integer range — the outlier channels that motivate LLM.int8() and SmoothQuant in trained
transformers, where a handful of activations sit at 20σ and drag the scale up for everyone else.
That is not the situation here. Q, K and V are Gaussian by construction, so at N·d = 65536 samples
the maximum lands at ~4.7σ, which is not an outlier but simply where the maximum of a sample that
size *is*. Clipping at 3.29σ shrinks the scale by only 1.4× — half a bit of resolution — and pays
for it with 66 saturated values whose error is unbounded, against a rounding error bounded at half
an LSB. The worst-case error degrades 4.6× for a relative-RMS improvement of nothing.

The second argument is methodological. Percentile calibration presumes a calibration set distinct
from the evaluation set. This project generates its tensors and quantizes them in the same breath,
so calibrating a percentile on the very tensor about to be quantized would be a claim that does not
survive scrutiny. Abs-max on the tensor itself is honest and describable in one line: *per-tensor
symmetric, abs-max calibrated.*

**Where the error actually ends up.** With the scheme complete, the three error sources named in
`quantize.py` can be separated by ablation (relative RMS, as a fraction of the FP32 output's σ):

| N, d | input quantization | + integer softmax | + UINT8 probabilities (full) |
|---|---|---|---|
| 128, 64 | 0.0146 | 0.0146 | 0.0158 |
| 256, 64 | 0.0162 | 0.0162 | 0.0179 |
| 512, 64 | 0.0179 | 0.0180 | 0.0209 |

**Rounding Q, K and V onto the INT8 grid accounts for ~90% of the total error.** The integer
softmax contributes nothing measurable — the `exp2` path is, to four decimal places, as accurate as
float `exp()` — and requantizing the probabilities to UINT8 adds the remaining ~10%. This settles
the obligation D-004 left open: a table sweep at fixed pipeline confirms it.

| N | 64 entries | 128 | **256** | 512 | 1024 |
|---|---|---|---|---|---|
| 128 | 0.0160 | 0.0158 | **0.0158** | 0.0158 | 0.0158 |
| 512 | 0.0210 | 0.0209 | **0.0209** | 0.0209 | 0.0208 |

The table stopped mattering below 128 entries. D-004's 256 is kept — it is already conservative,
still 512 bytes of LUTRAM, and growing it is provably pointless.

The practical consequence is that **the lever for more accuracy is the input format, not the
softmax.** If Phase 2 ever needs a tighter result, per-row or per-block scales for Q and K are the
place to spend the effort; a bigger table, a wider multiplier, or finer probabilities are all
optimizing the 10%.

**Alternatives rejected.** **Percentile clipping**, on the measurement above. **Power-of-two
scales** (`2^ceil(log2(absmax/127))`), which reduce requantization to a bare shift and need no
multiplier at all — rejected because D-004 and D-006 already commit the design to a 15-bit
multiplier in the exponent path, so the multiplier exists whether or not this uses it, while
rounding the scale up to the next power of two throws away up to a full bit of resolution.

**Consequence.** The scale depends on the tensor's own maximum, so it is a *dynamic*
per-invocation quantity: the host computes `absmax` before each call and derives the multiplier
from it. That is one reduction over N·d elements on the ARM core, negligible beside the attention
itself, but it does mean the score multiplier `M` is not baked into the bitstream — it is a
register the driver writes per call, which F2.4 has to expose.

---

## D-009 — The final normalization keeps 8 fractional bits

**Date:** 2026-09-06 · **Status:** accepted (refines the last stage of D-006)

**Decision.** The output normalization left-shifts the accumulator before dividing:

```
O_q = (O_acc << 8) // ℓ          # floor division, D-005
O   = O_q · (s_v / 256)          # one output LSB is s_v/256, not s_v
```

The emitted value needs 16 bits rather than 8. Everything else about D-006 stands: one division
per row, the factor of 255 still cancels, the numerator is still exact.

**Why.** D-006 specified `O = O_acc / ℓ`, which silently places the output on V's own INT8 grid
with `s_v` as its LSB. That is the wrong grid, and measurably so:

| N, d | fractional bits | relative RMS error |
|---|---|---|
| 128, 64 | 0 (as D-006 wrote it) | 0.132 |
| 128, 64 | 4 | 0.0175 |
| 128, 64 | **8** | **0.0158** |
| 128, 64 | 12 | 0.0158 |
| 256, 64 | 0 | 0.182 |
| 256, 64 | **8** | **0.0179** |
| 256, 64 | 16 | 0.0179 |

The reason is structural rather than a matter of tuning. Attention's second stage is a weighted
*average*: `O = Σpⱼvⱼ / Σpⱼ` with the weights summing to one. Averaging N values of a zero-centred
distribution shrinks the result toward the mean, so `|O|` scales roughly as `σ_v/√N_eff` while the
LSB stays pinned at `s_v`, which was calibrated against V's *maximum*. At N=256 the output occupies
about ±3 integer codes out of the 127 available. The error therefore *grows with N* — 0.132 at
N=128, 0.182 at N=256 — which is the signature of a format problem rather than a noise problem.

Eight bits is where the curve flattens: past it the residual error is the genuine quantization of
Q, K, V and the probabilities (D-008), and the division no longer contributes. Four bits gets most
of the way; twelve and sixteen are indistinguishable from eight.

**Cost.** Close to nothing. The shift is wiring, the divider gains 8 bits of numerator, and the
output widens from INT8 to INT16 on a tensor of N·d elements — against the N² that the score matrix
would cost, doubling the *output* is not where this design's bandwidth goes. `O_acc` was already
INT32 and had the headroom: the worst case at N=1024 is `1024 · 255 · 127 ≈ 3.3e7`, and shifting by
8 leaves it at 8.5e9 — which does **not** fit INT32, so the shifted numerator must be carried in 64
bits, or the division restructured. F0.6 has to size this explicitly rather than assume INT32
throughout.

**Alternatives rejected.** **Calibrating a separate output scale `s_o`** and requantizing to INT8
with `derive_multiplier` — this is what a production quantized network does, and it is more
accurate per bit emitted. Rejected for now because it needs an output calibration pass, which
introduces exactly the calibration-set problem D-008 avoided, and because the output tensor is
small enough that 16 bits costs less than the machinery would. Worth revisiting if Phase 2 chains
attention into a following quantized layer, where `s_o` would be that layer's input scale and comes
for free. **Leaving D-006 as written** and reporting a 13–18% error — rejected: an error that grows
with N would have discredited the N-sweep that is Phase 0's headline result.

**Consequence.** Two, one methodological and one still open.

This was found by measurement, not by review: D-006's normalization looked correct on paper and
reads correctly as algebra. The failure only appears once the *magnitudes* are traced through,
which is the argument for F0.3 producing a number before F0.6 starts building to the spec.

**Open:** D-006 describes the hardware realization as *one reciprocal per row and `d` multiplies*,
not `d` divisions. The reference implements a true floor division, which is the cleaner
specification but is not the same integer function as `round_shift(O_acc · ⌊2^k/ℓ⌋, k−8)` — a
reciprocal is itself quantized, and the two will disagree by an LSB on some elements. D-007 demands
the RTL reproduce the reference bit for bit, so one of the two has to give. The choice (pick a
reciprocal width `k` and adopt it in the reference, or make the hardware divide) belongs to F0.6,
where the accumulator widths are sized anyway. Naming it here so it is a decision rather than a
Phase 1 surprise.

---

<!--
Template for new entries:

## D-00N — One-line title

**Date:** YYYY-MM-DD · **Status:** proposed | accepted | superseded by D-00M

**Decision.** What was decided, concretely.

**Why.** The reasoning. Be specific: "it's faster" is useless, "it avoids N² writes to DRAM
because the K tile fits in BRAM" is not.

**Alternatives rejected.** What else was considered, and why not.

**Consequence.** What becomes harder or gets locked in by this decision.
-->
