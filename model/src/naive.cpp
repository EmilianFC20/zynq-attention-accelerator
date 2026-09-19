#include "common.hpp"
#include "dataflow.hpp"

namespace zaa {

// Steps to implement in F0.8:
//
//   1. S = QKᵀ  →  the array produces tiles of S; since the full S is N×N and doesn't fit in
//                  SRAM for large N, each tile is written out to DRAM as it comes.
//   2. row-wise softmax over S  →  second pass: re-read S from DRAM, apply the stable softmax
//                                  (subtract the row max), write it back.
//   3. O = P·V  →  third pass: re-read P from DRAM and multiply by V.
//
// The cost lives in steps 2 and 3: S is written once and read twice. That O(N²) traffic is
// exactly what the flash dataflow eliminates, and it is the number the project is chasing.
//
// Beware the trap: it is tempting to "optimize" the naive path by fusing passes or caching rows.
// Don't. The naive dataflow has to be the honest baseline of an accelerator that doesn't tile.
// If you half-improve it, the comparison stops meaning anything. If you want a third,
// intermediate option, add it as a separate dataflow with its own name.

DataflowResult NaiveDataflow::run(const Tensor&, const Tensor&, const Tensor&,
                                   const DatapathParams&) {
    throw NotImplemented("dataflow 'naive' — task F0.8 in docs/ROADMAP.md");
}

}  // namespace zaa
