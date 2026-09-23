// Tests for the simulator core: golden set loading, the checks that guard it, and the execution
// skeleton. The dataflows are still stubs, so nothing here compares numbers — that is F0.10. What
// is tested is that a mismatched configuration or a foreign vector set is refused by name.
//
// No test framework, on purpose: see DECISIONS.md D-002.

#include <cstdio>
#include <string>

#include "common.hpp"
#include "config.hpp"
#include "datapath.hpp"
#include "golden.hpp"
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
        test_skeleton();
    } catch (const std::exception& e) {
        std::printf("\nuncaught exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
