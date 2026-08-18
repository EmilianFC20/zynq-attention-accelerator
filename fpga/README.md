# Zynq integration — Phase 2

Empty for now. Starts in October 2026.

**This is the phase that turns the project from "RTL" into "ML systems".** The Zynq isn't a
standalone FPGA: it has two ARM Cortex-A9 cores running Linux attached to the programmable logic.
That makes it possible to expose the accelerator as a custom PyTorch operator and measure a real
end-to-end speedup against the CPU — which is a completely different story from "I wrote a
systolic array in Verilog", and points at the kind of role this project is aimed at.

## The path

```
PyTorch (custom op)  →  PYNQ overlay  →  AXI  →  accelerator in the PL  →  DDR3
        [ARM Cortex-A9 running Linux]              [programmable logic]
```

## Pending decision: PYNQ vs PetaLinux

**Recommendation: PYNQ.** Digilent publishes an official image for the Arty Z7-20. PYNQ loads
overlays (bitstream + metadata) directly from Python, which makes the bridge to the PyTorch
operator almost trivial. PetaLinux gives more control over the system but costs weeks of
configuration the schedule doesn't have.

When the decision is made, record it in `docs/DECISIONS.md`.

## Known risk: Vivado

Vivado isn't installed and is a ~100 GB download. **Start the installation in September**, not in
October when it's already needed. It's required to synthesize the bitstream even when using PYNQ.

## What will live here

- Vivado block design TCL scripts (version the TCL, not the binary project)
- PYNQ overlay (`.bit` + `.hwh`) and the Python control layer
- Custom PyTorch operator
- Benchmark scripts against the Cortex-A9 CPU baseline

## What has to be measured (F2.6)

Not just the speedup: the **breakdown** of where the time goes — DMA transfer, compute, driver
overhead. On small accelerators the cost of moving the data usually dominates the cost of
computing on it, and reporting that with data is more valuable than hiding it. A project that
only publishes the pretty number reads as marketing; one that explains where the time went reads
as engineering.
