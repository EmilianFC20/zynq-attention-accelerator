#include "sram.hpp"

#include "common.hpp"

namespace zaa {

SramModel::SramModel(uint64_t capacity_bytes) : capacity_bytes_(capacity_bytes) {
    if (capacity_bytes_ == 0) {
        throw ModelError("SramModel: capacity cannot be zero");
    }
}

void SramModel::allocate(const std::string&, uint64_t) {
    throw NotImplemented("SramModel::allocate — task F0.7 in docs/ROADMAP.md");
}

void SramModel::free(const std::string&) {
    throw NotImplemented("SramModel::free — task F0.7 in docs/ROADMAP.md");
}

void SramModel::reset() {
    used_bytes_ = 0;
    peak_bytes_ = 0;
}

}  // namespace zaa
