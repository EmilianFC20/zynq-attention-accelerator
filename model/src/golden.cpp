#include "golden.hpp"

#include <cstdio>
#include <vector>

#include "common.hpp"

namespace zaa {
namespace {

bool file_exists(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::fclose(f);
    return true;
}

std::string shape_str(const std::vector<uint32_t>& shape, Dtype dtype) {
    return Tensor(dtype, shape).describe();
}

// Reads one file of the set and requires an exact shape and dtype.
Tensor load_checked(const std::string& dir, const char* name, std::vector<uint32_t> shape,
                    Dtype dtype) {
    const std::string path = dir + "/" + name + ".bin";
    Tensor t = read_tensor(path);
    if (t.shape() != shape || t.dtype() != dtype) {
        throw ModelError(path + " is " + t.describe() + " but the config asks for " +
                         shape_str(shape, dtype));
    }
    return t;
}

}  // namespace

DatapathParams check_params(const Tensor& params, const std::string& origin) {
    if (params.dtype() != Dtype::Int32 || params.shape().size() != 1 ||
        params.shape()[0] < kNumParams) {
        throw ModelError(origin + " is " + params.describe() + ", expected at least (" +
                         std::to_string(kNumParams) + ") int32");
    }
    const int32_t* p = params.as_int32();

    struct Expected {
        const char* name;
        uint32_t index;
        int32_t value;
    };
    const Expected constants[] = {
        {"out_frac_bits", kParamOutFracBits, kOutFracBits},
        {"mult_bits", kParamMultBits, kMultBits},
        {"exp_frac_bits", kParamExpFracBits, kExpFracBits},
        {"exp_table_bits", kParamExpTableBits, kExpTableBits},
        {"exp_max_shift", kParamExpMaxShift, kExpMaxShift},
    };
    for (const Expected& c : constants) {
        if (p[c.index] != c.value) {
            throw ModelError(origin + ": " + c.name + " = " + std::to_string(p[c.index]) +
                             " but this model is built for " + std::to_string(c.value) +
                             " — the vectors were generated for a different datapath; regenerate "
                             "them or rebuild the model");
        }
    }

    DatapathParams out;
    out.score_multiplier = p[kParamScoreMultiplier];
    out.score_shift = p[kParamScoreShift];

    // M is an unsigned MULT_BITS-wide multiplier and the shift must be a legal shift of an int64
    // product. Anything else would be undefined behavior in the softmax stage, not a wrong answer.
    if (out.score_multiplier <= 0 || out.score_multiplier >= (int32_t{1} << kMultBits)) {
        throw ModelError(origin + ": score_multiplier " + std::to_string(out.score_multiplier) +
                         " does not fit in " + std::to_string(kMultBits) + " bits");
    }
    if (out.score_shift < 0 || out.score_shift >= 63) {
        throw ModelError(origin + ": score_shift " + std::to_string(out.score_shift) +
                         " is out of range");
    }
    return out;
}

GoldenSet load_golden_set(const Config& cfg) {
    const std::string& dir = cfg.vectors_dir;
    const uint32_t n = cfg.seq_len;
    const uint32_t d = cfg.head_dim;

    GoldenSet g;
    g.q = load_checked(dir, "q", {n, d}, Dtype::Int8);
    g.k = load_checked(dir, "k", {n, d}, Dtype::Int8);
    g.v = load_checked(dir, "v", {n, d}, Dtype::Int8);
    g.out = load_checked(dir, "out", {n, d}, Dtype::Int32);
    g.row_sums = load_checked(dir, "row_sums", {n, 1}, Dtype::Int32);

    const std::string params_path = dir + "/params.bin";
    g.params = check_params(read_tensor(params_path), params_path);

    const bool has_scores = file_exists(dir + "/scores.bin");
    const bool has_probs = file_exists(dir + "/probs.bin");
    if (has_scores != has_probs) {
        throw ModelError(dir + ": scores.bin and probs.bin must be present together — the set "
                               "is incomplete, regenerate it");
    }
    if (has_scores) {
        g.scores = load_checked(dir, "scores", {n, n}, Dtype::Int32);
        g.probs = load_checked(dir, "probs", {n, n}, Dtype::Int32);
    }
    return g;
}

}  // namespace zaa
