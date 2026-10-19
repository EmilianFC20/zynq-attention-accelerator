// Tests for the simulator core: golden set loading, the checks that guard it, the execution
// skeleton, and the PE array's arithmetic and timing model. The dataflows are still stubs, so no
// test here compares against the golden output — that is F0.10. What is tested is that a
// mismatched configuration or a foreign vector set is refused by name, and that the array computes
// exact integer products in the number of cycles D-014 says it should.
//
// No test framework, on purpose: see DECISIONS.md D-002.

#include <cstdio>
#include <initializer_list>
#include <string>

#include "common.hpp"
#include "config.hpp"
#include "datapath.hpp"
#include "golden.hpp"
#include "pe_array.hpp"
#include "simulator.hpp"
#include "stats.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (condition) {
        std::printf("  ok   %s\n", what.c_str());
    } else {
        std::printf("  FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

template <typename Exception, typename Fn>
void check_throws(Fn fn, const std::string& what) {
    ++g_checks;
    try {
        fn();
        std::printf("  FAIL %s (did not throw)\n", what.c_str());
        ++g_failures;
    } catch (const Exception&) {
        std::printf("  ok   %s\n", what.c_str());
    } catch (const std::exception& e) {
        std::printf("  FAIL %s (unexpected exception: %s)\n", what.c_str(), e.what());
        ++g_failures;
    }
}

zaa::Config smoke_config() { return zaa::load_config("../sweeps/smoke.cfg"); }

// A params tensor that matches this build, for the tests to corrupt one field at a time.
zaa::Tensor valid_params() {
    zaa::Tensor t(zaa::Dtype::Int32, {zaa::kNumParams});
    int32_t* p = reinterpret_cast<int32_t*>(t.raw().data());
    p[zaa::kParamScoreMultiplier] = 27559;
    p[zaa::kParamScoreShift] = 19;
    p[zaa::kParamOutFracBits] = zaa::kOutFracBits;
    p[zaa::kParamMultBits] = zaa::kMultBits;
    p[zaa::kParamExpFracBits] = zaa::kExpFracBits;
    p[zaa::kParamExpTableBits] = zaa::kExpTableBits;
    p[zaa::kParamExpMaxShift] = zaa::kExpMaxShift;
    return t;
}

void set_param(zaa::Tensor& t, uint32_t index, int32_t value) {
    reinterpret_cast<int32_t*>(t.raw().data())[index] = value;
}

void test_load_smoke() {
    std::printf("loading the smoke golden set\n");

    const zaa::GoldenSet g = zaa::load_golden_set(smoke_config());
    check(g.q.shape() == std::vector<uint32_t>{8, 8} && g.q.dtype() == zaa::Dtype::Int8,
          "q is (8, 8) int8");
    check(g.out.shape() == std::vector<uint32_t>{8, 8} && g.out.dtype() == zaa::Dtype::Int32,
          "out is (8, 8) int32");
    check(g.row_sums.shape() == std::vector<uint32_t>{8, 1}, "row_sums is (8, 1)");
    check(g.has_intermediates(), "smoke set carries scores and probs");
    check(g.params.score_multiplier > 0 && g.params.score_shift > 0,
          "score multiplier and shift were read");

    // A cross-file invariant of the reference: ℓ is the row sum of the UINT8 probabilities. If it
    // holds in C++, int32 payloads cross the Python→C++ boundary intact, not just int8 ones.
    const int32_t* probs = g.probs.as_int32();
    const int32_t* sums = g.row_sums.as_int32();
    bool consistent = true;
    for (uint32_t i = 0; i < 8; ++i) {
        int64_t acc = 0;
        for (uint32_t j = 0; j < 8; ++j) {
            const int32_t p = probs[i * 8 + j];
            consistent = consistent && p >= 0 && p <= 255;
            acc += p;
        }
        consistent = consistent && acc == sums[i];
    }
    check(consistent, "row_sums[i] == Σ probs[i, :], and every prob is in [0, 255]");
}

void test_config_mismatch() {
    std::printf("configuration and vectors that disagree\n");

    zaa::Config cfg = smoke_config();
    cfg.seq_len = 16;
    check_throws<zaa::ModelError>([&] { (void)zaa::load_golden_set(cfg); },
                                  "seq_len = 16 against the N=8 set throws");

    cfg = smoke_config();
    cfg.head_dim = 64;
    check_throws<zaa::ModelError>([&] { (void)zaa::load_golden_set(cfg); },
                                  "head_dim = 64 against the d=8 set throws");

    cfg = smoke_config();
    cfg.vectors_dir = "../vectors/does_not_exist";
    check_throws<zaa::ModelError>([&] { (void)zaa::load_golden_set(cfg); },
                                  "a missing vector directory throws");
}

void test_params() {
    std::printf("datapath constants: validation, not configuration\n");

    const zaa::DatapathParams p = zaa::check_params(valid_params(), "test");
    check(p.score_multiplier == 27559 && p.score_shift == 19, "a matching params tensor is accepted");

    const uint32_t constants[] = {zaa::kParamOutFracBits, zaa::kParamMultBits,
                                  zaa::kParamExpFracBits, zaa::kParamExpTableBits,
                                  zaa::kParamExpMaxShift};
    for (uint32_t index : constants) {
        zaa::Tensor t = valid_params();
        set_param(t, index, reinterpret_cast<const int32_t*>(t.raw().data())[index] + 1);
        check_throws<zaa::ModelError>([&] { (void)zaa::check_params(t, "test"); },
                                      "a foreign constant at index " + std::to_string(index) +
                                          " throws");
    }

    zaa::Tensor t = valid_params();
    set_param(t, zaa::kParamScoreMultiplier, 1 << zaa::kMultBits);
    check_throws<zaa::ModelError>([&] { (void)zaa::check_params(t, "test"); },
                                  "a multiplier wider than MULT_BITS throws");

    t = valid_params();
    set_param(t, zaa::kParamScoreShift, 64);
    check_throws<zaa::ModelError>([&] { (void)zaa::check_params(t, "test"); },
                                  "a shift of 64 throws instead of reaching undefined behavior");

    check_throws<zaa::ModelError>(
        [] { (void)zaa::check_params(zaa::Tensor(zaa::Dtype::Int32, {3}), "test"); },
        "a params tensor with too few fields throws");
    check_throws<zaa::ModelError>(
        [] { (void)zaa::check_params(zaa::Tensor(zaa::Dtype::Int8, {7}), "test"); },
        "a params tensor of the wrong dtype throws");
}

void test_stats_fold() {
    std::printf("folding the stats of sequential phases\n");

    // Two phases that run one after the other, like the naive dataflow's QKᵀ pass and its softmax
    // pass. The numbers are arbitrary; what matters is that each field is distinct.
    zaa::Stats pass1;
    pass1.cycles = 1000;
    pass1.dram_bytes_read = 16384;
    pass1.dram_bytes_written = 65536;
    pass1.mac_ops = 64000;
    pass1.pe_idle_cycles = 80000;
    pass1.sram_peak_bytes = 40960;

    zaa::Stats pass2;
    pass2.cycles = 400;
    pass2.dram_bytes_read = 65536;
    pass2.dram_bytes_written = 65536;
    pass2.mac_ops = 0;
    pass2.pe_idle_cycles = 57600;
    pass2.sram_peak_bytes = 30720;

    zaa::Stats total;
    total += pass1;  // inside operator+=: this = &total, later = pass1
    total += pass2;  // inside operator+=: this = &total, later = pass2

    check(total.cycles == 1400, "cycles accumulate");
    check(total.dram_bytes_read == 81920 && total.dram_bytes_written == 131072,
          "dram bytes accumulate");
    check(total.mac_ops == 64000, "mac ops accumulate");
    check(total.pe_idle_cycles == 137600, "pe idle cycles accumulate");
    check(total.sram_peak_bytes == 40960, "sram peak is the larger of the two, not the sum");
    check(pass1.cycles == 1000 && pass2.cycles == 400, "the folded-in phases are left untouched");
}

// A 2-D tensor from a row-major initializer list, for hand-computed examples.
template <typename T>
zaa::Tensor matrix(zaa::Dtype dtype, uint32_t rows, uint32_t cols, std::initializer_list<T> values) {
    zaa::Tensor t(dtype, {rows, cols});
    T* data = reinterpret_cast<T*>(t.raw().data());
    std::size_t i = 0;
    for (T v : values) {
        data[i++] = v;
    }
    return t;
}

bool equals(const zaa::Tensor& t, std::initializer_list<int32_t> values) {
    if (t.dtype() != zaa::Dtype::Int32 || t.numel() != values.size()) {
        return false;
    }
    std::size_t i = 0;
    for (int32_t v : values) {
        if (t.as_int32()[i++] != v) {
            return false;
        }
    }
    return true;
}

void test_pe_arithmetic() {
    std::printf("PE array arithmetic\n");
    using zaa::Dtype;

    zaa::PEArray pes(4, 4);
    const zaa::Tensor a = matrix<int8_t>(Dtype::Int8, 2, 2, {1, 2, 3, 4});
    const zaa::Tensor b = matrix<int8_t>(Dtype::Int8, 2, 2, {5, 6, 7, 8});
    const zaa::Tensor b_t = matrix<int8_t>(Dtype::Int8, 2, 2, {5, 7, 6, 8});

    check(equals(pes.matmul_int8(a, b, false), {19, 22, 43, 50}), "2x2 int8 product");
    check(equals(pes.matmul_int8(a, b_t, true), {19, 22, 43, 50}),
          "transpose_b reads b as (P, K), the layout of K in S = Q·Kᵀ");

    // D-006: the same byte is 200 as a probability and -56 as a query element.
    const zaa::Tensor p8 = matrix<uint8_t>(Dtype::UInt8, 1, 1, {200});
    const zaa::Tensor q8 = matrix<int8_t>(Dtype::Int8, 1, 1, {static_cast<int8_t>(200)});
    const zaa::Tensor v = matrix<int8_t>(Dtype::Int8, 1, 1, {-2});
    check(equals(pes.matmul_int8(p8, v, false), {-400}), "uint8 a is zero-extended");
    check(equals(pes.matmul_int8(q8, v, false), {112}), "int8 a is sign-extended");

    // Extreme operands: the 9x8 multiplier's corners.
    const zaa::Tensor a_min = matrix<int8_t>(Dtype::Int8, 1, 1, {-128});
    const zaa::Tensor p_max = matrix<uint8_t>(Dtype::UInt8, 1, 1, {255});
    const zaa::Tensor b_min = matrix<int8_t>(Dtype::Int8, 1, 1, {-128});
    check(equals(pes.matmul_int8(a_min, b_min, false), {16384}), "-128 x -128");
    check(equals(pes.matmul_int8(p_max, b_min, false), {-32640}), "255 x -128");

    check_throws<zaa::ModelError>([&] { (void)pes.matmul_int8(a, p8, false); },
                                  "a uint8 b operand is refused (the PE's B port is int8)");
    check_throws<zaa::ModelError>([&] { (void)pes.matmul_int8(a, v, false); },
                                  "mismatched inner dimensions are refused");

    // 255 x 127 accumulated 70000 times is 2.27e9, past INT32's 2.15e9.
    const uint32_t long_k = 70000;
    zaa::Tensor p_long(Dtype::UInt8, {1, long_k});
    zaa::Tensor v_long(Dtype::Int8, {long_k, 1});
    for (uint32_t i = 0; i < long_k; ++i) {
        p_long.raw()[i] = 255;
        v_long.raw()[i] = 127;
    }
    check_throws<zaa::ModelError>([&] { (void)pes.matmul_int8(p_long, v_long, false); },
                                  "an accumulator that leaves INT32 is an error, not a wrap");
}

void test_pe_timing() {
    std::printf("PE array timing model\n");

    // Per tile on a 4x4 array: compute K + 3 + 3, then 4 cycles of read-out.
    const zaa::PEArray serial(4, 4, false);
    const zaa::PEArray overlap(4, 4, true);

    check(serial.matmul_cycles(4, 5, 4) == 15, "one tile: K + (R-1) + (C-1) + R");
    check(overlap.matmul_cycles(4, 5, 4) == 15, "one tile: overlap has nothing to overlap with");

    // 8x8 output = 4 tiles, K = 8 (per-tile cost 18).
    check(serial.matmul_cycles(8, 8, 8) == 72, "no overlap: 4 tiles x 18");
    check(overlap.matmul_cycles(8, 8, 8) == 42, "overlap: a new tile every K cycles, 3 x 8 + 18");

    // K = 2 < R = 4: the next tile waits for the read-out, not the MACs (per-tile cost 12).
    check(serial.matmul_cycles(8, 2, 8) == 48, "short K, no overlap: 4 x 12");
    check(overlap.matmul_cycles(8, 2, 8) == 24, "short K, overlap: read-out bound, 3 x 4 + 12");

    // Partial tiles: a 5x7 output needs 2x2 tiles on a 4x4 array, and each one costs the full 13.
    zaa::PEArray pes(4, 4, false);
    const zaa::Tensor a(zaa::Dtype::Int8, {5, 3});
    const zaa::Tensor b(zaa::Dtype::Int8, {3, 7});
    (void)pes.matmul_int8(a, b, false);
    check(pes.stats().cycles == 52, "partial tiles cost as much as full ones: 4 x 13");
    check(pes.stats().mac_ops == 105, "useful MACs count only the real output: 5 x 7 x 3");
    check(pes.stats().pe_idle_cycles == 52 * 16 - 105, "every other PE-cycle is idle");
    check(pes.stats().pe_utilization(pes.num_pes()) <= 1.0, "utilization never exceeds 1");

    (void)pes.matmul_int8(a, b, false);
    check(pes.stats().cycles == 104, "a second matmul adds to the stats");
    pes.reset();
    check(pes.stats().cycles == 0 && pes.stats().mac_ops == 0, "reset clears the stats");

    check(!smoke_config().tile_overlap, "overlap is off unless a config turns it on");
    check(zaa::PEArray(smoke_config()).num_pes() == 16, "the array takes its size from the config");
}

void test_skeleton() {
    std::printf("execution skeleton\n");

    const zaa::Config cfg = smoke_config();
    const zaa::GoldenSet g = zaa::load_golden_set(cfg);
    // The dataflows are stubs until F0.8/F0.9. Reaching NotImplemented — rather than a ModelError
    // — proves the models were constructed and the dataflow was dispatched.
    check_throws<zaa::NotImplemented>([&] { (void)zaa::run_simulation(cfg, g); },
                                      "run_simulation reaches the dataflow");
}

}  // namespace

int main() {
    std::printf("=== test_core ===\n\n");

    try {
        test_load_smoke();
        test_config_mismatch();
        test_params();
        test_stats_fold();
        test_pe_arithmetic();
        test_pe_timing();
        test_skeleton();
    } catch (const std::exception& e) {
        std::printf("\nuncaught exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
