# Roadmap

Every task is sized at **2–6 hours**. Always take the first unchecked one.
When you complete one: check the box, move `▶ YOU ARE HERE`, write in `JOURNAL.md`, commit.

**Real time budget:**
- Aug 2–23, 2026: +20 h/week → ~65 h (before classes start on August 24)
- Sep 2026 onward: 5–10 h/week

**Critical date:** mid-October 2026 — new grad silicon openings are already posted by then and the
repo has to be presentable. A complete Phase 0 is already presentable.

---

## Phase 0 — Cycle-accurate C++ model (~65 h, target: Aug 23, 2026)

> A deliverable that stands on its own even if the RTL never arrives: a simulator validated
> against PyTorch, plus a design-space sweep with plots and one headline number.

- [x] **F0.1** — Repo scaffold, Python↔C++ I/O contract, git and public GitHub *(2 h)*
  - Folder structure, engineering docs, `.gitignore`
  - Binary vector format implemented and tested in both languages
  - `make` and `make test` working

- [x] **F0.2** — FP32 attention reference (`reference/attention_ref.py`) *(3 h)*
  - Single-head attention: `softmax(QKᵀ / √d) · V`, with no `nn.Module` dependencies
  - Validate against `torch.nn.functional.scaled_dot_product_attention`
  - Configurable in `N` and `d`; fixed seed for reproducibility

- [x] **F0.3** — INT8 quantization scheme (`reference/quantize.py`) *(4 h)*
  - Symmetric per-tensor scales for Q, K, V; INT32 accumulator
  - Decide and **document in DECISIONS.md** how softmax is handled: the exponential is not linear, so either it's done in fixed point with a LUT, or it gets requantized. This is the most important design decision of the phase — don't take it lightly
  - Measure the error against FP32 and set the tolerance that validation will use

- [x] **F0.4** — Full golden vector generation *(3 h)*
  - Extend `gen_vectors.py` to also dump the expected output and the scales
  - Vector sets: `smoke` (N=8, d=8), `small` (N=128, d=64), `full` (N=1024, d=64)
  - Document in `vectors/README.md` what each set contains

▶ **YOU ARE HERE**

- [ ] **F0.5** — Simulator core: config, tensors, execution skeleton *(4 h)*
  - Load golden vectors into the model and check shapes against the config
  - `Stats` struct with cycle, byte, and utilization counters

- [ ] **F0.6** — `PEArray`: R×C systolic array with cycle accounting *(6 h)*
  - INT8 MAC with INT32 accumulator
  - Systolic pipeline fill/drain model (it matters for small N)
  - Counters: total cycles, idle PE cycles, utilization

- [ ] **F0.7** — Memory models: `DramModel` and `SramModel` *(6 h)*
  - DRAM: configurable bandwidth and latency, counters for bytes read/written
  - SRAM: banks with an **explicit, hard capacity cap** — a tile that doesn't fit must be a
    detectable error, not a silent overflow. That cap is the central variable of the study
  - Calibrate the defaults against `docs/HARDWARE.md`

- [ ] **F0.8** — `naive` dataflow *(6 h)*
  - Materializes the full `S = QKᵀ`, writes it to DRAM, softmax in a second pass, then `S·V`
  - This is the baseline everything is measured against. It has to be a *fair* implementation,
    not a straw man: if you make it artificially bad, the result is worthless

- [ ] **F0.9** — `flash` dataflow with online softmax *(8 h)*
  - Blocked over K/V, never materializes `S`
  - Running max and sum with rescaling of the partial accumulators
  - The hardest task of the phase; the rescaling is where bugs slip in

- [ ] **F0.10** — Numerical validation against the golden vectors *(4 h)*
  - Both dataflows must match the INT8 reference **bit for bit** (D-007; the F0.3 measurement
    is an accuracy figure for `RESULTS.md`, not a pass threshold)
  - `make test` runs the full validation and fails loudly on a mismatch
  - **Without this, the project has no value.** It is literally the central argument

- [ ] **F0.11** — Sweep harness *(5 h)*
  - Sweep N ∈ {128, 256, 512, 1024, 2048, 4096}, d=64, array size, SRAM budget
  - Output to CSV so the plots are reproducible from raw data

- [ ] **F0.12** — Plots and analysis *(5 h)*
  - DRAM traffic vs N (both dataflows), cycles vs N, roofline, SRAM budget curve
  - Figures to `sweeps/figures/`, regenerable with a single command

- [ ] **F0.13** — `RESULTS.md` and public README *(5 h)*
  - The headline number at the very top, methodology below it
  - Hypothesis to confirm or refute: at d=64 and N=1024, naive moves ~2·N² bytes for `S` alone
    (~2 MB) against ~4·N·d (~256 KB) for flash → ~9× less traffic, with the gap growing with N.
    **If the real number is different, the real one gets reported.**

- [ ] **F0.14** — Slack and polish *(4 h)*
  - Buffer for whatever slipped. If there's time left over: profile the simulator or improve the
    README

---

## Phase 1 — Verified RTL (~81 h, Sep–Nov 2026)

> At 5–10 h/week. The goal is not a complete accelerator: it is **RTL that matches the model**.
> The RTL is synthesizable SystemVerilog, and the softmax unit gets a UVM testbench (D-011).

- [ ] **F1.1** — Install and smoke-test Verilator **5.x** *(2 h)*
  - The Ubuntu 22.04 package (4.038) is too old for SystemVerilog `interface`s and SVA; build
    from source or use a newer distro package
- [ ] **F1.2** — Single PE in SystemVerilog: INT8 MAC with INT32 accumulator + testbench *(4 h)*
  - Width constants in a shared `package`, not repeated per module
- [ ] **F1.3** — Parameterizable 12×12 systolic array *(8 h)*
- [ ] **F1.4** — On-chip buffers and tile-load state machine *(8 h)*
- [ ] **F1.5** — Fixed-point online softmax unit *(10 h)*
- [ ] **F1.5a** — SVA pass over the RTL written so far *(3 h)*
  - Handshake stability, no accumulator overflow, FSM legality, SRAM capacity cap
  - Every assertion must fire at least once on a deliberately broken input (an assertion that
    has never failed has never been tested)
- [ ] **F1.5b** — UVM environment for the softmax unit on xsim: agent and sequences *(8 h)*
  - Interface, driver, monitor, sequencer; directed sequences first, then constrained-random
  - Requires Vivado installed (pulls that Phase 2 gate forward)
- [ ] **F1.5c** — DPI-C scoreboard backed by the C++ model *(5 h)*
  - Expose the C++ softmax with a C ABI; the scoreboard compares bit for bit (D-007)
  - A single seeded mismatch must fail the run loudly
- [ ] **F1.5d** — Functional coverage and closure report *(5 h)*
  - Covergroups: running-max updates / rescale events, all-equal scores, `exp2` table
    underflow, D-009 rounding boundary, back-pressure
  - Measured coverage goes to `RESULTS.md` (R-8); unreachable bins are documented, not deleted
- [ ] **F1.6** — Full datapath integration *(8 h)*
- [ ] **F1.7** — Verilator co-simulation: RTL vs C++ model on the same vectors *(10 h)*
- [ ] **F1.8** — Cycle-accuracy match report and discrepancy analysis *(6 h)*
- [ ] **F1.9** — Vivado synthesis: resource and timing reports *(4 h)*

## Phase 2 — End-to-end Zynq integration (~50 h, Oct–Dec 2026)

> This is where the project stops being "RTL" and becomes "ML systems". It's the payoff.

- [ ] **F2.1** — Digilent PYNQ image for the Arty Z7-20, boot and verify *(4 h)*
- [ ] **F2.2** — Vivado block design: PS + accelerator over AXI *(8 h)*
- [ ] **F2.3** — DMA / AXI HP path to the DDR3 *(10 h)*
- [ ] **F2.4** — Python control layer on top of the PYNQ overlay *(6 h)*
- [ ] **F2.5** — Custom PyTorch operator that dispatches to the accelerator *(10 h)*
- [ ] **F2.6** — End-to-end measurement against a Cortex-A9 CPU baseline *(8 h)*
- [ ] **F2.7** — Correlation: does real silicon match what the Phase 0 model predicted? *(4 h)*

## Phase 3 — Dissemination (~30 h, Jan–Feb 2027)

- [ ] **F3.1** — Final README with figures and the end-to-end result *(8 h)*
- [ ] **F3.2** — Long-form technical post about the co-design process *(12 h)*
- [ ] **F3.3** — CV bullets and a 2-minute interview script *(4 h)*
- [ ] **F3.4** — Prepare answers to the obvious technical questions about the project *(6 h)*

---

## Known risks

| Risk | Mitigation |
|---|---|
| The semester eats all available time starting August 24 | Phase 0 is finished before then and is already presentable on its own |
| Vivado isn't installed and is a huge download (~100 GB) | Needed from F1.5b (xsim for UVM, D-011), not Phase 0. Start the download in September |
| Fixed-point softmax turns out harder than expected (F0.3, F1.5) | This is the real technical risk of the project. If it stalls, document the dead end — an honest analysis of why something is hard is technical signal too |
| The F1.7 co-simulation reveals large discrepancies | That *is* the result, not a failure. Analyze and report the cause |
