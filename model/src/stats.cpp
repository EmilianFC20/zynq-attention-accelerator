#include "stats.hpp"

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

std::string Stats::describe(uint64_t num_pes) const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "  cycles            " << cycles << "\n"
       << "  dram read         " << dram_bytes_read << " B\n"
       << "  dram written      " << dram_bytes_written << " B\n"
       << "  dram total        " << dram_bytes_total() << " B\n"
       << "  macs              " << mac_ops << "\n"
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
