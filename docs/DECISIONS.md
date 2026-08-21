# Design decisions

An ADR-style log: every non-obvious decision, with its reasoning and the alternatives rejected.

This file exists for a very concrete reason: in a technical interview the question won't be
*"what did you build?"* but ***"why did you build it that way?"***. The answers get written here,
at the moment the decision is made and the reasoning is fresh.

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

**Why.** The C++ model can't have external dependencies (see D-002), which rules out reading
`.npy` with a library or using Protobuf/HDF5. Writing a `.npy` parser by hand is possible, but
its header is a Python dict literal — you'd have to parse text in order to read a binary file,
which is absurd. Thirty-two bytes of fixed-width header are read with one `fread` and validated
with two `if`s.

**Alternatives rejected.** `.npy` (text header, needlessly fragile parser); CSV (loses
floating-point precision, enormous files at N=4096); HDF5 (heavy dependency).

**Consequence.** Any change to the format has to be made on both sides at once, and the `version`
field exists to catch the mismatch instead of silently reading garbage.

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
2. **Portability into the Phase 1 flow.** The same code will be linked against the Verilator
   harness for co-simulation. Fewer dependencies, less friction at that point.
3. `cmake` isn't even installed on the development machine, and a 30-line Makefile more than
   covers a project this size.

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

**Date:** 2026-08-19 · **Status:** accepted (table size to be revisited in F0.3 once the
quantization error is measured)

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

**Alternatives rejected.** **Round-half-to-even** (banker's rounding, what IEEE 754 does) is
unbiased even on exact ties, but requires tie detection plus an LSB inspection: more logic, and
three more opportunities for the implementations to differ subtly. Ties are rare in this datapath.
The gemmlowp/TFLite requantization path makes the same trade. **Truncation** is rejected on the
bias argument above.

**Consequence.** The one place this rule does not reach is the final normalization
`O = O_acc / ℓ`, which is a true division rather than a shift, and whose numerator is signed
because `V` is signed. C++ integer division truncates toward zero while Python's `//` floors
toward −∞ — they disagree on negative operands. The division must therefore be specified as an
explicit formula and implemented through a shared floor-division helper on both sides. This is the
most likely single source of a Phase 1 co-simulation mismatch, and it is cheap to prevent now and
expensive to find later.

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
