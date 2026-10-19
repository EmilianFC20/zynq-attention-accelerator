#pragma once

// R×C output-stationary systolic array (D-014). Each PE owns one element of the output tile and
// keeps its INT32 accumulator in place while the operands stream past: rows of A enter from the
// left edge, columns of B from the top edge, each skewed by one cycle per row or column so that
// matching operands meet in the right PE.
//
// The multiplier is 9-bit signed × 8-bit signed (D-006). The A operand is either INT8 (Q, for
// S = Q·Kᵀ) or UINT8 zero-extended to 9 bits (the probabilities, for O_acc = P·V); B is always
// INT8. One PE design serves both of attention's matmuls.

#include <cstdint>

#include "config.hpp"
#include "stats.hpp"
#include "vector_io.hpp"

namespace zaa {

class PEArray {
public:
    PEArray(uint32_t rows, uint32_t cols, bool tile_overlap = false);
    explicit PEArray(const Config& cfg)
        : PEArray(cfg.array_rows, cfg.array_cols, cfg.tile_overlap) {}

    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t num_pes() const { return rows_ * cols_; }
    bool tile_overlap() const { return tile_overlap_; }

    // out = a · b, or a · bᵀ when `transpose_b` is set. `a` is (M, K) int8 or uint8; `b` is
    // (K, P) int8, or (P, K) when transposed. The result is (M, P) int32.
    //
    // The arithmetic is exact: every product is accumulated into an INT32 register, and a sum that
    // leaves the INT32 range throws ModelError. The hardware would wrap silently; the model
    // refuses to, so an undersized accumulator surfaces as an error rather than a wrong answer.
    //
    // Adds the cycles, useful MACs and idle PE-cycles of the operation to `stats()`.
    Tensor matmul_int8(const Tensor& a, const Tensor& b, bool transpose_b);

    // Cycles the array spends on an (M, K)·(K, P) matmul. Separated from the arithmetic so the
    // dataflows can cost a schedule without computing it, and so the timing model can be tested
    // on its own. The model is described next to its implementation in pe_array.cpp.
    uint64_t matmul_cycles(uint64_t m, uint64_t k, uint64_t p) const;

    const Stats& stats() const { return stats_; }
    void reset();

private:
    uint32_t rows_;
    uint32_t cols_;
    bool tile_overlap_;
    Stats stats_;
};

}  // namespace zaa
