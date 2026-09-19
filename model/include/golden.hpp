#pragma once

// A golden vector set, loaded and checked against the configuration it is about to be run with.
// The file layout is documented in vectors/README.md.
//
// Every check here exists because the alternative is a silent mismatch: a config that says N=1024
// pointed at the N=128 vectors, or vectors generated for a different fixed-point datapath. Both
// produce plausible numbers that are wrong, and both are far cheaper to catch at load time than to
// debug from a failed comparison in F0.10.

#include <string>

#include "config.hpp"
#include "datapath.hpp"
#include "vector_io.hpp"

namespace zaa {

struct GoldenSet {
    // Inputs: what the accelerator receives.
    Tensor q;  // (N, d) int8
    Tensor k;  // (N, d) int8
    Tensor v;  // (N, d) int8
    DatapathParams params;

    // Expected results: what a correct accelerator emits. Never handed to a dataflow.
    Tensor out;       // (N, d) int32 — 16-bit values, D-009
    Tensor row_sums;  // (N, 1) int32 — softmax denominators ℓ

    // Stage outputs, for localizing a mismatch. Absent when the set was generated with
    // --no-intermediates; both are present or neither is.
    Tensor scores;  // (N, N) int32 — q8 · k8ᵀ
    Tensor probs;   // (N, N) int32 — UINT8-valued probabilities

    bool has_intermediates() const { return !scores.shape().empty(); }
};

// Loads the set from cfg.vectors_dir. Throws ModelError on a missing file, a shape or dtype that
// disagrees with the config, or datapath constants that disagree with datapath.hpp.
GoldenSet load_golden_set(const Config& cfg);

// Checks a params.bin tensor against the compiled-in datapath and returns the per-set values.
// Exposed separately so the check can be tested without a file on disk.
DatapathParams check_params(const Tensor& params, const std::string& origin);

}  // namespace zaa
