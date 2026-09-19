#include "common.hpp"
#include "dataflow.hpp"

namespace zaa {

// Online softmax, to be implemented in F0.9.
//
// The idea: walk K and V in blocks. For each block of Q rows, three things are kept live in
// SRAM — the running max `m`, the running sum `l`, and the output accumulator `O` — and the full
// S matrix is never materialized.
//
// For each K/V block j:
//
//   S_j     = Q_i · K_jᵀ                        (tile, fits in SRAM)
//   m_new   = max(m, rowmax(S_j))
//   P_j     = exp(S_j - m_new)
//   l       = l · exp(m - m_new) + rowsum(P_j)   ← rescaling of the sum
//   O       = O · exp(m - m_new) + P_j · V_j     ← rescaling of the accumulator
//   m       = m_new
//
// and at the end O is divided by l.
//
// **The rescaling is where everything breaks.** Both `exp(m - m_new)` factors have to be applied
// to the sum and to the output accumulator before adding the new contribution. Forgetting either
// one produces a result that looks almost correct for short sequences — where the max barely
// moves — and degrades silently as N grows. It's a bug that passes the N=8 smoke test and fails
// at N=1024. That is why the three vector sets from F0.4 exist.
//
// Suggestion: implement and validate in FP32 first, confirm against the reference, and **only
// then** add quantization. Debugging the rescaling and the fixed point at the same time means
// fighting two sources of error at once.
//
// On block size: it comes out of the SRAM budget. Let `sram_.fits()` decide the largest block
// that fits, rather than hard-coding it — that way the F0.11 SRAM budget sweep moves the real
// variable and not just a decorative parameter.

DataflowResult FlashDataflow::run(const Tensor&, const Tensor&, const Tensor&,
                                   const DatapathParams&) {
    throw NotImplemented("dataflow 'flash' — task F0.9 in docs/ROADMAP.md");
}

}  // namespace zaa
