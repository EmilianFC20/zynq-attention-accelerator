#pragma once

// Fixed-point widths of the attention datapath. These are the hardware specification: the RTL of
// Phase 1 hardcodes the same values, and the model must not be more flexible than the silicon.
//
// Every golden vector set carries its own copy of these numbers in params.bin. They are checked
// against the constants below and a disagreement is an error — the set was generated for a
// different datapath, and adopting its values would model hardware that does not exist.
// See docs/DECISIONS.md D-010.

#include <cstdint>

namespace zaa {

constexpr int32_t kOutFracBits = 8;    // fractional bits kept through the final division, D-009
constexpr int32_t kMultBits = 15;      // width of the requantization multiplier M
constexpr int32_t kExpFracBits = 8;    // exponent fraction = exp2 table index width, D-004
constexpr int32_t kExpTableBits = 15;  // table entries are UQ1.15, D-004
constexpr int32_t kExpMaxShift = 16;   // exponents below 2^-16 are clamped to zero, D-004

// Positional layout of params.bin, mirrored from PARAM_FIELDS in reference/gen_vectors.py.
// Fields are appended, never inserted.
enum ParamIndex : uint32_t {
    kParamScoreMultiplier = 0,
    kParamScoreShift = 1,
    kParamOutFracBits = 2,
    kParamMultBits = 3,
    kParamExpFracBits = 4,
    kParamExpTableBits = 5,
    kParamExpMaxShift = 6,
    kNumParams = 7,
};

// The per-set values the datapath is configured with: the (M, shift) pair that carries an INT32
// score into the exponent domain. Derived on the host from the quantization scales, which is what
// the ARM core does in Phase 2; the fabric only ever sees these two integers.
struct DatapathParams {
    int32_t score_multiplier = 0;
    int32_t score_shift = 0;
};

}  // namespace zaa
