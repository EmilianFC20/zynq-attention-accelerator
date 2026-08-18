#pragma once

// The two dataflows the project compares, running on exactly the same array, DRAM, and SRAM
// infrastructure. Sharing that infrastructure is what makes the comparison fair.
//
// Status: STUBS — `naive` is F0.8, `flash` is F0.9.

#include <memory>
#include <string>

#include "config.hpp"
#include "dram.hpp"
#include "pe_array.hpp"
#include "sram.hpp"
#include "stats.hpp"
#include "vector_io.hpp"

namespace zaa {

struct DataflowResult {
    Tensor output;  // (N, d) int8
    Stats stats;
};

class Dataflow {
public:
    Dataflow(const Config& cfg, DramModel& dram, SramModel& sram, PEArray& pes)
        : cfg_(cfg), dram_(dram), sram_(sram), pes_(pes) {}
    virtual ~Dataflow() = default;

    virtual const char* name() const = 0;
    virtual DataflowResult run(const Tensor& q, const Tensor& k, const Tensor& v) = 0;

protected:
    const Config& cfg_;
    DramModel& dram_;
    SramModel& sram_;
    PEArray& pes_;
};

// Baseline: materializes the full S = QKᵀ in DRAM, does softmax in a second pass, then S·V.
// It has to be a **fair** implementation: if it is loaded with artificial ballast, the comparison
// is worthless and anyone reading the code will notice.
class NaiveDataflow : public Dataflow {
public:
    using Dataflow::Dataflow;
    const char* name() const override { return "naive"; }
    DataflowResult run(const Tensor& q, const Tensor& k, const Tensor& v) override;
};

// Blocked over K/V with online softmax: never materializes S. The running max and sum are updated
// as it goes, and the partial accumulators are rescaled whenever a new max appears.
// That rescaling is where the bugs get in — it's worth testing on its own before integrating it.
class FlashDataflow : public Dataflow {
public:
    using Dataflow::Dataflow;
    const char* name() const override { return "flash"; }
    DataflowResult run(const Tensor& q, const Tensor& k, const Tensor& v) override;
};

// Throws ModelError if the name is neither "naive" nor "flash".
std::unique_ptr<Dataflow> make_dataflow(const std::string& name, const Config& cfg, DramModel& dram,
                                        SramModel& sram, PEArray& pes);

}  // namespace zaa
