#pragma once

#include <stdexcept>
#include <string>

namespace zaa {

// Thrown from the stubs that aren't implemented yet. main() catches it and reports it as a
// pending roadmap item rather than as a crash, so the scaffold runs clean from day one.
struct NotImplemented : std::runtime_error {
    explicit NotImplemented(const std::string& what)
        : std::runtime_error("not implemented: " + what) {}
};

// Configuration errors, I/O errors, or resource-constraint violations (for example, a tile that
// does not fit in the SRAM budget).
struct ModelError : std::runtime_error {
    explicit ModelError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace zaa
