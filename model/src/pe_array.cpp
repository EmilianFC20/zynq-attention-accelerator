#include "pe_array.hpp"

#include "common.hpp"

namespace zaa {

PEArray::PEArray(uint32_t rows, uint32_t cols) : rows_(rows), cols_(cols) {
    if (rows_ == 0 || cols_ == 0) {
        throw ModelError("PEArray: dimensions must be greater than zero");
    }
}

Tensor PEArray::matmul_int8(const Tensor&, const Tensor&, bool) {
    throw NotImplemented("PEArray::matmul_int8 — task F0.6 in docs/ROADMAP.md");
}

void PEArray::reset() { stats_ = Stats{}; }

}  // namespace zaa
