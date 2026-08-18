# Platform: Digilent Arty Z7-20

Every default in the simulator is calibrated against this board. If a number in the model can't
be traced back to this page, it's an invented assumption and it has to be flagged as such.

## The chip: Zynq XC7Z020-1CLG400C

It isn't an FPGA: it's an **SoC**. Two halves on the same die, connected by AXI.

### PS — Processing System (the "hard" half)

| | |
|---|---|
| CPU | Dual-core ARM Cortex-A9 @ 650 MHz |
| DRAM | 512 MB DDR3, 16-bit bus, 1050 MT/s |
| Theoretical peak bandwidth | 1050 MT/s × 2 bytes ≈ **2.1 GB/s** |

The DDR3 hangs off the PS. The PL reaches it through the **AXI HP** (High Performance) ports,
which are 64 bits wide.

### PL — Programmable Logic (the "soft" half, an Artix-7)

| Resource | Count |
|---|---|
| Logic slices | 13,300 |
| 6-input LUTs | 53,200 |
| Flip-flops | 106,400 |
| Block RAM | **630 KB** (140 blocks of 36 Kb) |
| DSP48E1 slices | **220** |

## The two numbers that govern the design

### 630 KB of BRAM

This is the total on-chip memory budget, and it is **the central variable of the whole Phase 0
study**. The reason flash attention tiling exists is exactly this: the score matrix `S = QKᵀ` is
N×N, and at N=1024 in INT8 that's 1 MB — **it doesn't fit**. Not even using the chip's entire
BRAM for nothing else.

Without tiling, `S` has to go out to the DDR3 and come back. With tiling, it is never
materialized. The F0.11 sweep should show exactly where in the (N, SRAM budget) space the regime
changes.

Note: 630 KB is the chip total. The realistic budget for data buffers is lower, because control
logic and AXI FIFOs consume BRAM too. Use ~500 KB as the working figure and document the
assumption.

### 220 DSP slices

This sets the size of the systolic array. See `DECISIONS.md` D-003: 12×12 fits directly, 16×16
requires INT8 packing.

## Bandwidth conversion for the model

The model counts in **bytes per PL cycle**, not in GB/s. At a PL clock of 100 MHz (conservative
and easy to close timing on):

```
2.1 GB/s ÷ 100 MHz ≈ 21 bytes/cycle
```

That's the default for `dram_bw_bytes_per_cycle` in the configs. It is a **theoretical peak**:
real DDR3 efficiency with non-ideal access patterns runs somewhere between 60 % and 80 %, and the
AXI HP port imposes its own overhead. When the real Phase 2 measurements arrive, this number gets
recalibrated against what was observed — and that comparison (predicted vs measured) is itself
one of the most interesting results the project can produce.

The default latency, `dram_latency_cycles = 30`, is an order-of-magnitude estimate for a DDR3
access through AXI. Also pending recalibration in Phase 2.

## Other board interfaces

HDMI in/out, microSD (PYNQ/Linux boot), 4 Pmod connectors, Arduino shield-compatible header,
built-in USB-JTAG/UART. None of this is relevant to the accelerator, except the microSD, which is
where the PYNQ image boots from in Phase 2.

## Sources

- [Arty Z7 — Digilent Reference](https://digilent.com/reference/programmable-logic/arty-z7/start)
- [Arty Z7 Reference Manual (PDF)](https://media.digikey.com/pdf/Data%20Sheets/Digilent%20PDFs/Arty_Z7_RM_Web.pdf)
- Xilinx DS190 — Zynq-7000 SoC Data Sheet: Overview
- Xilinx WP486 — Deep Learning with INT8 Optimization on Xilinx Devices (INT8 packing on DSP48E1)
