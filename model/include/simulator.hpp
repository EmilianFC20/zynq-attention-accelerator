#pragma once

// Execution skeleton: builds the hardware models from a configuration, runs one dataflow on a
// golden set's inputs, and checks that what comes back honors the output contract.
//
// Numerical comparison against the expected output is deliberately not done here — that is F0.10,
// and it belongs to the test suite, not to every simulator run.

#include "config.hpp"
#include "dataflow.hpp"
#include "golden.hpp"

namespace zaa {

// Runs cfg.dataflow on the inputs of `golden`. Each call starts from freshly constructed models,
// so two runs never share counters. Throws ModelError if the dataflow's output is not (N, d) int32.
DataflowResult run_simulation(const Config& cfg, const GoldenSet& golden);

}  // namespace zaa
