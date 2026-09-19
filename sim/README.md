# Co-simulation — Phase 1

Empty for now. This is where the thing that verifies the RTL matches the C++ model will live.

**This directory contains the project's central result.** Not the accelerator: the *proof* that
the accelerator does what the model predicted. A testbench written by the same person who wrote
the RTL shares their misconceptions; a comparison against an independent model, written earlier
and validated against PyTorch, does not.

## Plan

Verilator compiles the RTL to C++ and links it against the same harness the architectural model
already uses. Both consume the same golden vectors and their outputs are compared along two axes:

1. **Numerical** — does the RTL produce the same bits as the model?
2. **Temporal** — does the cycle count match, and if not, where do the differences come from?

The second axis is the interesting one. It is expected that the model won't be exactly right;
what matters is **being able to explain the difference**. That analysis goes to
`docs/RESULTS.md` (R-5) and is probably what will get discussed most in an interview.

## Requirements

Verilator 5.x or later (task F1.1). The RTL is SystemVerilog (D-011), which the 4.x releases
shipped by older distributions do not handle well:

```bash
verilator --version   # must report 5.x or later
```

The UVM testbench for the softmax unit (tasks F1.5b–F1.5d) runs on the Vivado simulator (xsim),
not Verilator.
