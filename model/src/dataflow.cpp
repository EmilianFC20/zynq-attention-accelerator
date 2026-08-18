#include "dataflow.hpp"

#include "common.hpp"

namespace zaa {

std::unique_ptr<Dataflow> make_dataflow(const std::string& name, const Config& cfg, DramModel& dram,
                                        SramModel& sram, PEArray& pes) {
    if (name == "naive") {
        return std::make_unique<NaiveDataflow>(cfg, dram, sram, pes);
    }
    if (name == "flash") {
        return std::make_unique<FlashDataflow>(cfg, dram, sram, pes);
    }
    throw ModelError("unknown dataflow: '" + name + "' (expected 'naive' or 'flash')");
}

}  // namespace zaa
