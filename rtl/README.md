# RTL — Phase 1

Empty for now. Phase 1 starts in September 2026, after the Phase 0 C++ model is validated.

**The order matters and is not negotiable:** the purpose of the RTL in this phase is not "build an
accelerator", it is **build an accelerator that matches the model**. Writing RTL before there
is a validated model to compare it against throws away the project's central argument, because
nothing independent would be left to verify it against.

## What will live here

- `accel_pkg.sv` — shared width constants (D-004, D-006, D-009), see D-011
- `pe.sv` — processing element: INT8 MAC with INT32 accumulator
- `systolic_array.sv` — parameterizable array (12×12 baseline, see DECISIONS.md D-003)
- `softmax_unit.sv` — fixed-point online softmax
- `tile_buffer.sv` — on-chip buffers and load state machine
- `attention_top.sv` — full datapath integration

## Conventions (D-011)

- Synthesizable SystemVerilog only; no assertions or simulation constructs in this directory
  (SVA checkers live in `sim/` and are attached with `bind`).
- No `interface`s. A stream is a packed-struct payload plus separate `valid` / `ready` ports,
  named AXI-Stream style: `*_tdata`, `*_tvalid`, `*_tready`.

## Resource budget (XC7Z020)

| Resource | Available | Target |
|---|---|---|
| DSP48E1 | 220 | 144 (12×12 array) + softmax and control |
| BRAM | 630 KB | ~500 KB for data |
| LUTs | 53,200 | to be determined |

Details in `docs/HARDWARE.md`. Tasks in `docs/ROADMAP.md` (F1.x).
