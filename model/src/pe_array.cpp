#include "pe_array.hpp"

#include <algorithm>
#include <limits>
#include <string>

#include "common.hpp"

namespace zaa {
namespace {

uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// The shape of a 2-D operand, or an error naming which operand and why.
void require_matrix(const Tensor& t, const char* name) {
    if (t.shape().size() != 2) {
        throw ModelError(std::string("PEArray: ") + name + " must be 2-D, got " + t.describe());
    }
}

}  // namespace

PEArray::PEArray(uint32_t rows, uint32_t cols, bool tile_overlap)
    : rows_(rows), cols_(cols), tile_overlap_(tile_overlap) {
    if (rows_ == 0 || cols_ == 0) {
        throw ModelError("PEArray: dimensions must be greater than zero");
    }
}

// Timing model (D-014). An (M, K)·(K, P) matmul is cut into T = ⌈M/R⌉·⌈P/C⌉ output tiles, each
// computed by the whole array. A partial tile at the edge of the output costs exactly as much as a
// full one: the PEs that fall outside the matrix still sit in the pipeline, doing nothing useful.
//
// One tile, start to finish, is three phases back to back:
//
//   compute   PE (i, j) receives its first operand pair i + j cycles after PE (0, 0) and then
//             does K MACs, so the last PE, (R-1, C-1), finishes K + (R-1) + (C-1) cycles after
//             the tile starts. The R+C-2 extra cycles are the fill and drain of the wavefront.
//   read-out  the finished accumulators shift down the columns, one row per cycle, so C results
//             leave the bottom edge per cycle and the whole tile is out after R cycles.
//
// Without overlap (tile_overlap_ == false), the next tile starts only after the previous one's
// read-out ends: every tile pays compute + read-out in full.
//
// With overlap, each PE copies its finished sum into a shadow register and starts the next tile
// immediately, so a new tile can start every K cycles — but the shadow registers of tile n are
// only free once tile n's read-out (R cycles) is over. If K < R, the next tile has to wait for
// the read-out, not the MACs. Only the last tile pays its fill, drain and read-out in full.
uint64_t PEArray::matmul_cycles(uint64_t m, uint64_t k, uint64_t p) const {
    const uint64_t tiles = ceil_div(m, rows_) * ceil_div(p, cols_);
    const uint64_t r = rows_;
    const uint64_t c = cols_;

    if (tiles == 0) {
        return 0;
    }

    // Compute (K MACs plus the R+C-2 cycles of wavefront fill and drain), then R cycles of read-out.
    const uint64_t one_tile = k + (r - 1) + (c - 1) + r;
    if (!tile_overlap_) {
        return tiles * one_tile;
    }

    // A new tile every max(K, R) cycles; only the last one is paid in full.
    return (tiles - 1) * std::max(k, r) + one_tile;
}

Tensor PEArray::matmul_int8(const Tensor& a, const Tensor& b, bool transpose_b) {
    require_matrix(a, "a");
    require_matrix(b, "b");

    // D-006: the multiplier is 9-bit signed × 8-bit signed. A may be INT8 or UINT8 (zero-extended
    // to 9 bits); B only INT8. Anything else would need a PE this design doesn't have.
    if (a.dtype() != Dtype::Int8 && a.dtype() != Dtype::UInt8) {
        throw ModelError("PEArray: a must be int8 or uint8, got " + a.describe());
    }
    if (b.dtype() != Dtype::Int8) {
        throw ModelError("PEArray: b must be int8, got " + b.describe());
    }

    const uint32_t m = a.shape()[0];
    const uint32_t k = a.shape()[1];
    const uint32_t b_k = transpose_b ? b.shape()[1] : b.shape()[0];
    const uint32_t p = transpose_b ? b.shape()[0] : b.shape()[1];
    if (b_k != k) {
        throw ModelError("PEArray: inner dimensions disagree: a is " + a.describe() + ", b is " +
                         b.describe() + (transpose_b ? " (transposed)" : ""));
    }

    // Signedness is decided once, from the dtype — the same byte 0xC8 is 200 as a probability and
    // -56 as a query element (vector_io.hpp).
    const bool a_unsigned = a.dtype() == Dtype::UInt8;
    const uint8_t* a_raw = a.raw().data();
    const int8_t* b_data = b.as_int8();
    auto a_at = [&](uint32_t i, uint32_t kk) -> int32_t {
        const uint8_t byte = a_raw[static_cast<std::size_t>(i) * k + kk];
        return a_unsigned ? static_cast<int32_t>(byte) : static_cast<int32_t>(static_cast<int8_t>(byte));
    };
    auto b_at = [&](uint32_t kk, uint32_t j) -> int32_t {
        const std::size_t idx = transpose_b ? static_cast<std::size_t>(j) * k + kk
                                            : static_cast<std::size_t>(kk) * p + j;
        return b_data[idx];
    };

    Tensor out(Dtype::Int32, {m, p});
    int32_t* out_data = out.as_int32_mut();

    // The functional result does not depend on the tile order, so it is computed directly. The
    // accumulation runs in 64 bits only to *detect* an overflow of the 32-bit hardware register.
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            int64_t acc = 0;
            for (uint32_t kk = 0; kk < k; ++kk) {
                acc += static_cast<int64_t>(a_at(i, kk)) * b_at(kk, j);
            }
            if (acc > std::numeric_limits<int32_t>::max() ||
                acc < std::numeric_limits<int32_t>::min()) {
                throw ModelError("PEArray: accumulator overflow at (" + std::to_string(i) + ", " +
                                 std::to_string(j) + "): " + std::to_string(acc) +
                                 " does not fit INT32 (K = " + std::to_string(k) + ")");
            }
            out_data[static_cast<std::size_t>(i) * p + j] = static_cast<int32_t>(acc);
        }
    }

    // Every PE-cycle the array was busy either did a useful MAC or was wasted: wavefront fill and
    // drain, read-out, or a PE outside the edge of a partial tile.
    const uint64_t cycles = matmul_cycles(m, k, p);
    const uint64_t macs = static_cast<uint64_t>(m) * k * p;
    stats_.cycles += cycles;
    stats_.mac_ops += macs;
    stats_.pe_idle_cycles += cycles * num_pes() - macs;

    return out;
}

void PEArray::reset() { stats_ = Stats{}; }

}  // namespace zaa
