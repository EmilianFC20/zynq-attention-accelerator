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
