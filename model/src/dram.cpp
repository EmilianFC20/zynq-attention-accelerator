#include "dram.hpp"

#include "common.hpp"

namespace zaa {

DramModel::DramModel(uint32_t bytes_per_cycle, uint32_t latency_cycles)
    : bytes_per_cycle_(bytes_per_cycle), latency_cycles_(latency_cycles) {
    if (bytes_per_cycle_ == 0) {
        throw ModelError("DramModel: bytes_per_cycle cannot be zero");
    }
}

uint64_t DramModel::read(uint64_t) {
    throw NotImplemented("DramModel::read — task F0.7 in docs/ROADMAP.md");
}

uint64_t DramModel::write(uint64_t) {
    throw NotImplemented("DramModel::write — task F0.7 in docs/ROADMAP.md");
}

void DramModel::reset() { stats_ = Stats{}; }

}  // namespace zaa
