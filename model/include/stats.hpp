#pragma once

// The counters a simulator run produces. They are the entire product of Phase 0: the F0.12 plots
// and the numbers in RESULTS.md all come from here.

#include <cstdint>
#include <string>

namespace zaa {

struct Stats {
    // Time
    uint64_t cycles = 0;

    // External memory traffic. The central number of the project: the thesis is that tiling with
    // online softmax cuts dram_bytes_read + dram_bytes_written from O(N²) down to O(N·d).
    uint64_t dram_bytes_read = 0;
    uint64_t dram_bytes_written = 0;

    // Systolic array occupancy
    uint64_t mac_ops = 0;         // useful multiply-accumulates
    uint64_t pe_idle_cycles = 0;  // wasted PE-cycles (fill, drain, partial tiles)

    // Pressure on the on-chip memory
    uint64_t sram_peak_bytes = 0;

    uint64_t dram_bytes_total() const { return dram_bytes_read + dram_bytes_written; }

    // Folds in the counters of a phase that runs *after* this one — how the naive dataflow's
    // three passes compose into one result. It is not a way to combine components that run
    // concurrently: overlap between DRAM and compute is the dataflow's schedule to account for.
    Stats& operator+=(const Stats& later);

    // Useful MACs / (cycles × available PEs). 1.0 would be a perfectly occupied array.
    double pe_utilization(uint64_t num_pes) const;

    // Arithmetic intensity in MACs per byte moved: the X coordinate of the roofline.
    double arithmetic_intensity() const;

    std::string describe(uint64_t num_pes) const;

    // One CSV line, for the F0.11 sweep harness.
    static std::string csv_header();
    std::string to_csv_row() const;
};

}  // namespace zaa
