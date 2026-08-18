# Results

**Hard rule: this file only receives numbers that came out of running something.**
Anything not yet measured says `TBD`. Never an estimate dressed up as a measurement — a made-up
number that survives until an interview is a far bigger risk than an empty cell.

Every result must include how to reproduce it.

---

## Status: Phase 0 in progress (F0.1 complete)

Nothing has been measured yet. The tables below are ready to be filled in.

---

## R-1 — DRAM traffic: naive vs flash

*Task: F0.11 / F0.12 · Status: TBD*

Configuration: d = 64, INT8, SRAM budget = TBD, array = TBD.

| N | naive (bytes) | flash (bytes) | reduction |
|---|---|---|---|
| 128 | TBD | TBD | TBD |
| 256 | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD |
| 2048 | TBD | TBD | TBD |
| 4096 | TBD | TBD | TBD |

**Hypothesis to confirm or refute (F0.13):** naive moves ~2·N² bytes for the `S` matrix alone
(writing it and reading it back) against ~4·N·d for flash (Q, K, V, O once each). For N=1024 and
d=64 that would be ~2 MB against ~256 KB, i.e. ~9×, with the gap growing linearly with N.

If the measured number turns out different, **the measured one gets reported and the difference
gets explained**. An honest analysis of why the simple model failed to predict reality is better
interview material than a round number that lines up suspiciously well with theory.

## R-2 — Array cycles and utilization

*Task: F0.11 / F0.12 · Status: TBD*

| N | dataflow | cycles | PE utilization | memory-bound? |
|---|---|---|---|---|
| | | TBD | | |

## R-3 — SRAM budget curve

*Task: F0.11 / F0.12 · Status: TBD*

Past how much on-chip memory does tiling stop mattering? This is the question that connects
directly to the XC7Z020's real 630 KB BRAM budget.

## R-4 — Synthesis resources (Vivado)

*Task: F1.9 · Status: TBD*

| Resource | Used | Available | % |
|---|---|---|---|
| DSP48E1 | TBD | 220 | |
| LUTs | TBD | 53,200 | |
| BRAM (KB) | TBD | 630 | |
| Fmax | TBD | | |

## R-5 — RTL vs C++ model agreement

*Task: F1.7 / F1.8 · Status: TBD*

The central result of the project. How close did the model's cycle prediction land to the real
RTL, and what accounts for the discrepancies?

## R-6 — End-to-end on the Zynq: accelerator vs Cortex-A9 CPU

*Task: F2.6 · Status: TBD*

| N | A9 CPU (ms) | Accelerator (ms) | speedup |
|---|---|---|---|
| | TBD | | |

Include the breakdown of where the time goes: DMA transfer, compute, driver overhead. On small
accelerators, transfer overhead usually dominates, and saying so with data is more valuable than
hiding it.

## R-7 — Predicted vs measured

*Task: F2.7 · Status: TBD*

Closing the loop: the Phase 0 model predicted X, the silicon delivered Y. This comparison is what
turns the project into a co-design story instead of three projects glued together.
