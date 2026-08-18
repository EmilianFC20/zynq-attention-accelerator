#pragma once

// R×C systolic array of processing elements. Each PE does one INT8 MAC with an INT32 accumulator.
//
// Status: STUB — implement in F0.6.

#include <cstdint>

#include "config.hpp"
#include "stats.hpp"
#include "vector_io.hpp"

namespace zaa {

class PEArray {
public:
    PEArray(uint32_t rows, uint32_t cols);
    explicit PEArray(const Config& cfg) : PEArray(cfg.array_rows, cfg.array_cols) {}

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t num_pes() const { return rows_ * cols_; }

    // INT8 matrix multiply with INT32 accumulation, mapped onto the array.
    // Returns the result and accumulates the cycles spent into `stats()`.
    //
    // To implement in F0.6. What has to be modeled and is easy to forget:
    //   - **Pipeline fill and drain.** An R×C systolic array takes ~R+C cycles to fill before it
    //     produces the first result. For small tiles that cost dominates, and it is exactly what
    //     makes a larger array not always win.
    //   - **Partial tiles.** When a dimension isn't a multiple of the array size, the leftover
    //     PEs do useless work. Those cycles go into `pe_idle_cycles`, they don't get ignored:
    //     they are the difference between theoretical and real utilization.
    //   - The cycle accounting has to be defensible against the Phase 1 RTL. If the model says
    //     1000 cycles and the Verilog says 1400, the F1.7 co-simulation will demand an
    //     explanation for where those 400 came from.
    Tensor matmul_int8(const Tensor& a, const Tensor& b, bool transpose_b);

    const Stats& stats() const { return stats_; }
    void reset();

private:
    uint32_t rows_;
    uint32_t cols_;
    Stats stats_;
};

}  // namespace zaa
