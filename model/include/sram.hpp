#pragma once

// Model of the on-chip memory (BRAM), with an **explicit, hard capacity cap**.
//
// Status: STUB — implement in F0.7.
//
// This is the conceptually most important component of the simulator. The SRAM budget is the
// independent variable of the entire study: the whole reason flash attention tiling exists is
// that the N×N S matrix does not fit in on-chip memory. If this model let you allocate more than
// exists, the flash dataflow would lose its advantage and the project would say nothing.
//
// That is why `allocate` must **fail loudly** when capacity is exceeded, never grow silently. A
// tile that doesn't fit is a design error the simulator has to catch.

#include <cstdint>
#include <string>

#include "config.hpp"

namespace zaa {

class SramModel {
public:
    explicit SramModel(uint64_t capacity_bytes);
    explicit SramModel(const Config& cfg) : SramModel(cfg.sram_bytes) {}

    // Allocates a named buffer. Throws ModelError if it doesn't fit — that error is the
    // interesting result, not a nuisance to be silenced.
    void allocate(const std::string& name, uint64_t bytes);
    void free(const std::string& name);

    uint64_t capacity_bytes() const { return capacity_bytes_; }
    uint64_t used_bytes() const { return used_bytes_; }
    uint64_t peak_bytes() const { return peak_bytes_; }
    uint64_t available_bytes() const { return capacity_bytes_ - used_bytes_; }

    // Does a tile of this size fit? Used by the dataflows to size their tiling.
    bool fits(uint64_t bytes) const { return bytes <= available_bytes(); }

    void reset();

private:
    uint64_t capacity_bytes_;
    uint64_t used_bytes_ = 0;
    uint64_t peak_bytes_ = 0;
};

}  // namespace zaa
