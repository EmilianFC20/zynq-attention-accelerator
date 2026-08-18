// Cycle-accurate simulator for the attention accelerator.
//
//   ./build/accel_sim --config ../sweeps/smoke.cfg
//   ./build/accel_sim --config ../sweeps/smoke.cfg --dataflow naive
//
// Status: the loading pipeline (config + vectors + shape validation) already works.
// The simulation engine is stubs; see docs/ROADMAP.md.

#include <cstdio>
#include <cstring>
#include <string>

#include "common.hpp"
#include "config.hpp"
#include "dataflow.hpp"
#include "dram.hpp"
#include "pe_array.hpp"
#include "sram.hpp"
#include "vector_io.hpp"

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

// Loads q/k/v and checks that their shapes match what the configuration says. A silent mismatch
// between vectors and config is one of the most expensive bugs to find later.
void load_and_check(const zaa::Config& cfg, zaa::Tensor& q, zaa::Tensor& k, zaa::Tensor& v) {
    const std::string dir = cfg.vectors_dir;
    q = zaa::read_tensor(dir + "/q.bin");
    k = zaa::read_tensor(dir + "/k.bin");
    v = zaa::read_tensor(dir + "/v.bin");

    const zaa::Tensor* tensors[] = {&q, &k, &v};
    const char* names[] = {"q", "k", "v"};

    for (int i = 0; i < 3; ++i) {
        const zaa::Tensor& t = *tensors[i];
        if (t.shape().size() != 2 || t.shape()[0] != cfg.seq_len || t.shape()[1] != cfg.head_dim) {
            throw zaa::ModelError(std::string(names[i]) + ".bin has shape " + t.describe() +
                                  " but the config asks for (" + std::to_string(cfg.seq_len) + ", " +
                                  std::to_string(cfg.head_dim) + ") int8");
        }
        if (t.dtype() != zaa::Dtype::Int8) {
            throw zaa::ModelError(std::string(names[i]) + ".bin must be int8, it is " + t.describe());
        }
    }
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

        zaa::Tensor q, k, v;
        load_and_check(cfg, q, k, v);
        std::printf("vectors loaded from %s\n  q %s\n  k %s\n  v %s\n\n", cfg.vectors_dir.c_str(),
                    q.describe().c_str(), k.describe().c_str(), v.describe().c_str());

        zaa::DramModel dram(cfg);
        zaa::SramModel sram(cfg);
        zaa::PEArray pes(cfg);
        auto dataflow = zaa::make_dataflow(cfg.dataflow, cfg, dram, sram, pes);

        std::printf("running dataflow '%s'...\n", dataflow->name());
        const zaa::DataflowResult result = dataflow->run(q, k, v);

        std::printf("\nresults\n%s\n", result.stats.describe(pes.num_pes()).c_str());
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
