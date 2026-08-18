# Design-space sweeps

Simulator configurations and the scripts that generate the Phase 0 plots.

## Current configurations

| File | What it represents |
|---|---|
| `smoke.cfg` | N=8, d=8, 4×4 array. Only for checking that the toolchain runs |
| `baseline.cfg` | N=1024, d=64, 12×12 array. The project's target design point |

Run one:

```bash
cd model
./build/accel_sim --config ../sweeps/smoke.cfg
./build/accel_sim --config ../sweeps/smoke.cfg --dataflow naive   # override the dataflow
```

## Format

Plain text, `key = value`, one per line, `#` comments to end of line. The valid keys are in
`model/include/config.hpp`; an unknown key is an error, not silently ignored — a typo in a
parameter name that gets ignored produces an entire sweep run at the default values without
anyone noticing.

## Pending (F0.11 / F0.12)

- `run_sweep.py` — sweeps N, array size, and SRAM budget across both dataflows, and writes a CSV
  to `raw/`.
- `plot.py` — generates the figures into `figures/` from the CSV.

The separation between raw data and plots is deliberate: the small CSVs are versioned so the
figures are auditable and regenerable without re-running the whole sweep. `raw/` is in the
`.gitignore` for the large runs.

Planned figures:

1. **DRAM traffic vs N**, both dataflows, log-log scale. The project's headline plot.
2. **Cycles vs N**, with array utilization annotated.
3. **Roofline** — arithmetic intensity against attainable performance, showing where each
   dataflow stops being memory-bound.
4. **SRAM budget curve** — past how much on-chip memory tiling stops mattering. This is the one
   that connects the study to the XC7Z020's real 630 KB.
