#include "simulator.hpp"

#include <algorithm>

#include "common.hpp"
#include "dram.hpp"
#include "pe_array.hpp"
#include "sram.hpp"

namespace zaa {

DataflowResult run_simulation(const Config& cfg, const GoldenSet& golden) {
    DramModel dram(cfg);
    SramModel sram(cfg);
    PEArray pes(cfg);
    auto dataflow = make_dataflow(cfg.dataflow, cfg, dram, sram, pes);

    DataflowResult result = dataflow->run(golden.q, golden.k, golden.v, golden.params);

    const std::vector<uint32_t> expected_shape = {cfg.seq_len, cfg.head_dim};
    if (result.output.shape() != expected_shape || result.output.dtype() != Dtype::Int32) {
        throw ModelError(std::string("dataflow '") + dataflow->name() + "' returned " +
                         result.output.describe() + ", the output contract is (" +
                         std::to_string(cfg.seq_len) + ", " + std::to_string(cfg.head_dim) +
                         ") int32");
    }

    // The SRAM model is the authority on on-chip pressure, whatever the dataflow reported.
    result.stats.sram_peak_bytes = std::max(result.stats.sram_peak_bytes, sram.peak_bytes());
    return result;
}

}  // namespace zaa
