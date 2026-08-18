#pragma once

// Model of the external DDR3: bandwidth and latency, and above all traffic counters.
//
// Status: STUB — implement in F0.7.

#include <cstdint>

#include "config.hpp"
#include "stats.hpp"

namespace zaa {

class DramModel {
public:
    DramModel(uint32_t bytes_per_cycle, uint32_t latency_cycles);

    explicit DramModel(const Config& cfg)
        : DramModel(cfg.dram_bw_bytes_per_cycle, cfg.dram_latency_cycles) {}

    // Model a read/write burst: accumulate bytes and return the cycles it costs.
    //
    // To implement in F0.7. Considerations worth keeping in view:
    //   - Latency amortizes over a large burst but dominates on small accesses. A model that
    //     merely divides bytes by bandwidth will be far too optimistic precisely for the naive
    //     dataflow, which makes many small accesses to the S matrix. That would bias the
    //     comparison in favor of the baseline and ruin the result.
    //   - Overlap (several bursts in flight) can be modeled or not. Decide explicitly and record
    //     it in DECISIONS.md; real AXI hardware does overlap.
    uint64_t read(uint64_t bytes);
    uint64_t write(uint64_t bytes);

    const Stats& stats() const { return stats_; }
    void reset();

private:
    uint32_t bytes_per_cycle_;
    uint32_t latency_cycles_;
    Stats stats_;
};

}  // namespace zaa
