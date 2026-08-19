# zynq-attention-accelerator

**An INT8 attention accelerator, co-designed from a cycle-accurate C++ model down to RTL on a Zynq-7000 SoC and verified to match at every step.**

> **Status: Phase 0 in progress (started August 2026).** The architectural model is being built now. No performance numbers are published yet, and none will be until they are measured. See
> [`docs/RESULTS.md`](docs/RESULTS.md) for what has and hasn't been quantified.

---

## What this is

Attention is memory-bound. The `N×N` score matrix `S = QKᵀ` doesn't fit in on-chip memory — at
`N=1024` in INT8 it is 1 MB, and the Zynq XC7Z020 on this board has **630 KB of block RAM
total**. So either `S` goes to external DRAM and comes back twice, or the computation is tiled so
that `S` is never materialized at all.

That tradeoff is the entire reason FlashAttention exists, and a small FPGA with real DDR3 attached
is an unusually honest place to study it: the memory wall isn't a rounding error here, it's the
dominant term.

This project builds both dataflows on the same modeled hardware and measures the difference —
then implements the winner in RTL and runs it for real.

## Why it's structured this way

Most FPGA accelerator projects are a Verilog systolic array with a testbench. This one is built
around a claim that is harder to make and worth more:

**every layer is verified against the one above it.**

```
   PyTorch reference (FP32 → INT8)          ← definition of "correct"
            ↓  validated against
   C++ cycle-accurate model                 ← predicts cycles and DRAM traffic
            ↓  co-simulated against
   Verilog RTL                              ← must match the model, cycle for cycle
            ↓  deployed as
   PyTorch custom op on the Zynq            ← measured against the model's prediction
```

A model nobody checked is a hypothesis. RTL checked only against its author's own testbench
inherits its author's misconceptions. The interesting engineering — and the part that resembles
actual accelerator work — is in the seams between those layers, not in any one of them.

The last step closes the loop: the Zynq's hard ARM cores run Linux, so the accelerator is reachable
from Python as a custom PyTorch operator. That makes the final comparison an end-to-end one —
predicted vs. measured on real silicon — rather than a simulation result.

## Platform

Digilent **Arty Z7-20** — Zynq **XC7Z020** SoC.

| | |
|---|---|
| PS | Dual-core ARM Cortex-A9 @ 650 MHz |
| PL | 53,200 LUTs · **220 DSP48E1** · **630 KB BRAM** |
| DRAM | 512 MB DDR3, 16-bit @ 1050 MT/s (≈ 2.1 GB/s peak) |

Full specs and how they constrain the design: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Repository layout

| Path | Contents |
|---|---|
| `reference/` | PyTorch golden reference and vector generation |
| `model/` | Cycle-accurate C++ simulator — no external dependencies |
| `sweeps/` | Design-space sweep configs, plotting scripts, figures |
| `rtl/` | Verilog accelerator *(Phase 1)* |
| `sim/` | Verilator co-simulation against the C++ model *(Phase 1)* |
| `fpga/` | Vivado project, PYNQ overlay, PyTorch operator *(Phase 2)* |
| `docs/` | Roadmap, engineering journal, design decisions, results |

## Building

The C++ model needs nothing but a C++17 compiler and `make` — no cmake, no third-party libraries.
That is [a deliberate decision](docs/DECISIONS.md), not an accident.

```bash
cd model
make          # build the simulator
make test     # generate smoke vectors and run the test suite
make run      # run the simulator on the smoke configuration
```

Python (PyTorch + NumPy) is used only for the golden reference, vector generation, and plots.

## Roadmap

| Phase | Deliverable | Target |
|---|---|---|
| **0** | Cycle-accurate C++ model, validated against PyTorch, with a design-space sweep | Aug 2026 |
| **1** | Verilog RTL, co-simulated against the model | Sep–Oct 2026 |
| **2** | End-to-end on the Zynq: PYNQ overlay, PyTorch custom op, measured speedup | Oct–Dec 2026 |
| **3** | Write-up and analysis | Jan–Feb 2027 |

Task-level detail, including what is done and what isn't:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Author

**Emiliano Fernández Cervantes** — M.S. Computer Engineering, University of Southern California.

[emilian.website](https://emilian.website) ·
[LinkedIn](https://www.linkedin.com/in/emiliano-fernandez-cervantes/) ·
[GitHub](https://github.com/EmilianFC20)
