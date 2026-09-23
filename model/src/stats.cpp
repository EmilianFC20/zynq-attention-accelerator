// The simulator's scoreboard. A run does not produce attention outputs for their own sake; it
// produces measurements of how the hardware would behave, and this file turns the raw counters in
// Stats into those measurements:
//   - pe_utilization:       useful MACs / (cycles × PEs) — how busy the array was
//   - arithmetic_intensity: MACs per DRAM byte — the roofline X coordinate
//   - operator+=:           combines the Stats of phases that run one after another
//   - describe / to_csv_row: human-readable summary and one CSV row for the sweeps

#include "stats.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace zaa {

double Stats::pe_utilization(uint64_t num_pes) const {
    if (cycles == 0 || num_pes == 0) {
        return 0.0;
    }
    return static_cast<double>(mac_ops) / static_cast<double>(cycles * num_pes);
}

double Stats::arithmetic_intensity() const {
    const uint64_t bytes = dram_bytes_total();
    if (bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(mac_ops) / static_cast<double>(bytes);
}

Stats& Stats::operator+=(const Stats& later) {
    // Totals accumulate: the second phase's time, traffic, and work come on top of the first's.
    cycles += later.cycles;
    dram_bytes_read += later.dram_bytes_read;
    dram_bytes_written += later.dram_bytes_written;
    mac_ops += later.mac_ops;
    pe_idle_cycles += later.pe_idle_cycles;

    // A peak does not: the phases never hold SRAM at the same time, so the run's peak is the
    // larger of the two, not their sum.
    sram_peak_bytes = std::max(sram_peak_bytes, later.sram_peak_bytes);
    return *this;
}

std::string Stats::describe(uint64_t num_pes) const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "  cycles            " << cycles << "\n"
       << "  dram read         " << dram_bytes_read << " B\n"
       << "  dram written      " << dram_bytes_written << " B\n"
       << "  dram total        " << dram_bytes_total() << " B\n"
       << "  macs              " << mac_ops << "\n"
       << "  pe idle cycles    " << pe_idle_cycles << "\n"
       << "  pe utilization    " << pe_utilization(num_pes) << "\n"
       << "  arith. intensity  " << arithmetic_intensity() << " MAC/B\n"
       << "  sram peak         " << sram_peak_bytes << " B";
    return os.str();
}

std::string Stats::csv_header() {
    return "cycles,dram_bytes_read,dram_bytes_written,dram_bytes_total,mac_ops,pe_idle_cycles,"
           "sram_peak_bytes";
}

std::string Stats::to_csv_row() const {
    std::ostringstream os;
    os << cycles << ',' << dram_bytes_read << ',' << dram_bytes_written << ',' << dram_bytes_total()
       << ',' << mac_ops << ',' << pe_idle_cycles << ',' << sram_peak_bytes;
    return os.str();
}

}  // namespace zaa
