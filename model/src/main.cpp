// Cycle-accurate simulator for the attention accelerator.
//
//   ./build/accel_sim --config ../sweeps/smoke.cfg
//   ./build/accel_sim --config ../sweeps/smoke.cfg --dataflow naive
//
// Status: the loading pipeline (config + full golden set + shape and datapath validation) works.
// The simulation engine is stubs; see docs/ROADMAP.md.

#include <cstdio>
#include <cstring>
#include <string>

#include "common.hpp"
#include "config.hpp"
#include "golden.hpp"
#include "simulator.hpp"

namespace {

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s --config PATH [options]\n"
        "\n"
        "  --config PATH      configuration file (see sweeps/smoke.cfg)\n"
        "  --dataflow NAME    override the dataflow: 'naive' or 'flash'\n"
        "  --vectors PATH     override the golden vector directory\n"
        "  --help             this help\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string dataflow_override;
    std::string vectors_override;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_next = (i + 1) < argc;

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--config" && has_next) {
            config_path = argv[++i];
        } else if (arg == "--dataflow" && has_next) {
            dataflow_override = argv[++i];
        } else if (arg == "--vectors" && has_next) {
            vectors_override = argv[++i];
        } else {
            std::fprintf(stderr, "unrecognized argument: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (config_path.empty()) {
        std::fprintf(stderr, "missing --config\n\n");
        print_usage(argv[0]);
        return 2;
    }

    try {
        zaa::Config cfg = zaa::load_config(config_path);
        if (!dataflow_override.empty()) {
            cfg.dataflow = dataflow_override;
        }
        if (!vectors_override.empty()) {
            cfg.vectors_dir = vectors_override;
        }

        std::printf("configuration (%s)\n%s\n\n", config_path.c_str(), cfg.describe().c_str());

        const zaa::GoldenSet golden = zaa::load_golden_set(cfg);
        std::printf("golden set loaded from %s\n", cfg.vectors_dir.c_str());
        std::printf("  inputs    q %s   k %s   v %s\n", golden.q.describe().c_str(),
                    golden.k.describe().c_str(), golden.v.describe().c_str());
        std::printf("  expected  out %s   row_sums %s\n", golden.out.describe().c_str(),
                    golden.row_sums.describe().c_str());
        std::printf("  datapath  M=%d  shift=%d  (constants match this build)\n",
                    golden.params.score_multiplier, golden.params.score_shift);
        std::printf("  stages    %s\n\n", golden.has_intermediates()
                                               ? "scores and probs present"
                                               : "absent (generated with --no-intermediates)");

        std::printf("running dataflow '%s'...\n", cfg.dataflow.c_str());
        const zaa::DataflowResult result = zaa::run_simulation(cfg, golden);
        const uint64_t num_pes = uint64_t{cfg.array_rows} * cfg.array_cols;
        std::printf("\nresults\n%s\n", result.stats.describe(num_pes).c_str());
        return 0;

    } catch (const zaa::NotImplemented& e) {
        // Expected while the engine is a stub. The scaffold has to run clean: config and vector
        // loading were already exercised above, which is what this binary can verify today.
        std::printf("\n[pending] %s\n", e.what());
        std::printf(
            "\nThe scaffold works: configuration and golden vectors were loaded and validated\n"
            "correctly. The simulation engine gets implemented following docs/ROADMAP.md.\n");
        return 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
