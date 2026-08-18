#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "common.hpp"

namespace zaa {
namespace {

std::string trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Strips end-of-line comments (everything after '#').
std::string strip_comment(const std::string& s) {
    const std::size_t hash = s.find('#');
    return hash == std::string::npos ? s : s.substr(0, hash);
}

uint64_t parse_u64(const std::string& key, const std::string& value) {
    try {
        const long long parsed = std::stoll(value);
        if (parsed < 0) {
            throw ModelError(key + ": cannot be negative (" + value + ")");
        }
        return static_cast<uint64_t>(parsed);
    } catch (const std::invalid_argument&) {
        throw ModelError(key + ": expected an integer, got '" + value + "'");
    } catch (const std::out_of_range&) {
        throw ModelError(key + ": value out of range (" + value + ")");
    }
}

}  // namespace

std::string Config::describe() const {
    std::ostringstream os;
    os << "  problem      N=" << seq_len << "  d=" << head_dim << "\n"
       << "  array        " << array_rows << "x" << array_cols << " PEs ("
       << (array_rows * array_cols) << " MACs)\n"
       << "  sram         " << sram_bytes << " bytes (" << (sram_bytes / 1024) << " KB)\n"
       << "  dram         " << dram_bw_bytes_per_cycle << " B/cycle, latency "
       << dram_latency_cycles << " cycles\n"
       << "  dataflow     " << dataflow << "\n"
       << "  vectors      " << vectors_dir;
    return os.str();
}

Config load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw ModelError("could not open configuration: " + path);
    }

    Config cfg;
    std::string line;
    int lineno = 0;

    while (std::getline(in, line)) {
        ++lineno;
        const std::string content = trim(strip_comment(line));
        if (content.empty()) {
            continue;
        }

        const std::size_t eq = content.find('=');
        if (eq == std::string::npos) {
            throw ModelError(path + ":" + std::to_string(lineno) +
                             ": expected 'key = value', got '" + content + "'");
        }

        const std::string key = trim(content.substr(0, eq));
        const std::string value = trim(content.substr(eq + 1));
        if (value.empty()) {
            throw ModelError(path + ":" + std::to_string(lineno) + ": '" + key + "' has no value");
        }

        if (key == "seq_len") {
            cfg.seq_len = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "head_dim") {
            cfg.head_dim = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "array_rows") {
            cfg.array_rows = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "array_cols") {
            cfg.array_cols = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "sram_bytes") {
            cfg.sram_bytes = parse_u64(key, value);
        } else if (key == "dram_bw_bytes_per_cycle") {
            cfg.dram_bw_bytes_per_cycle = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "dram_latency_cycles") {
            cfg.dram_latency_cycles = static_cast<uint32_t>(parse_u64(key, value));
        } else if (key == "dataflow") {
            cfg.dataflow = value;
        } else if (key == "vectors_dir") {
            cfg.vectors_dir = value;
        } else {
            throw ModelError(path + ":" + std::to_string(lineno) + ": unknown key '" + key + "'");
        }
    }

    if (cfg.seq_len == 0 || cfg.head_dim == 0) {
        throw ModelError(path + ": seq_len and head_dim must be greater than zero");
    }
    if (cfg.array_rows == 0 || cfg.array_cols == 0) {
        throw ModelError(path + ": array dimensions must be greater than zero");
    }
    if (cfg.dram_bw_bytes_per_cycle == 0) {
        throw ModelError(path + ": dram_bw_bytes_per_cycle cannot be zero");
    }
    if (cfg.dataflow != "naive" && cfg.dataflow != "flash") {
        throw ModelError(path + ": dataflow must be 'naive' or 'flash', got '" + cfg.dataflow + "'");
    }

    return cfg;
}

}  // namespace zaa
