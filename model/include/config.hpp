#pragma once

// Simulator configuration, read from a `key = value` text file.
// The files live in sweeps/. See sweeps/smoke.cfg for a commented example.

#include <cstdint>
#include <string>

namespace zaa {

struct Config {
    // Problem
    uint32_t seq_len = 8;   // N
    uint32_t head_dim = 8;  // d

    // Systolic array
    uint32_t array_rows = 4;
    uint32_t array_cols = 4;

    // On-chip memory. The working default is 500 KB, not the XC7Z020's full 630 KB: control
    // logic and AXI FIFOs consume BRAM too. See docs/HARDWARE.md.
    uint64_t sram_bytes = 512000;

    // DRAM. 21 B/cycle ≈ 2.1 GB/s from the Arty Z7's DDR3 at a 100 MHz PL clock.
    // A theoretical peak, pending recalibration against real Phase 2 measurements.
    uint32_t dram_bw_bytes_per_cycle = 21;
    uint32_t dram_latency_cycles = 30;

    // "naive" | "flash"
    std::string dataflow = "flash";

    // Directory holding q.bin / k.bin / v.bin, relative to the working directory.
    std::string vectors_dir = "../vectors/smoke";

    // Prints the configuration as a readable block.
    std::string describe() const;
};

// Throws ModelError if the file doesn't exist, has an unknown key, or an invalid value.
Config load_config(const std::string& path);

}  // namespace zaa
