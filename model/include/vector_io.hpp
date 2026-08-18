#pragma once

// Reader/writer for the ZAAV golden vector binary format.
// An exact mirror of reference/vector_io.py. The spec lives in docs/DECISIONS.md (D-001).
//
//   offset  size    field
//   0       4       magic "ZAAV"
//   4       4       version  (uint32 LE)
//   8       4       dtype    (uint32 LE)
//   12      4       ndim     (uint32 LE, <= 4)
//   16      16      dims[4]  (uint32 LE x4)
//   32      ...     raw C-contiguous little-endian data
//
// Any change to the format goes on both sides at once and bumps the version number.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zaa {

constexpr uint32_t kVectorFormatVersion = 1;
constexpr std::size_t kVectorHeaderSize = 32;
constexpr std::size_t kMaxDims = 4;

enum class Dtype : uint32_t {
    Int8 = 0,
    Int32 = 1,
    Float32 = 2,
};

std::size_t dtype_size(Dtype dt);
const char* dtype_name(Dtype dt);

class Tensor {
public:
    Tensor() = default;
    Tensor(Dtype dtype, std::vector<uint32_t> shape);

    Dtype dtype() const { return dtype_; }
    const std::vector<uint32_t>& shape() const { return shape_; }
    std::size_t numel() const;
    std::size_t size_bytes() const { return raw_.size(); }

    std::vector<uint8_t>& raw() { return raw_; }
    const std::vector<uint8_t>& raw() const { return raw_; }

    // Typed accessors. They throw ModelError if the dtype doesn't match.
    const int8_t* as_int8() const;
    const int32_t* as_int32() const;
    const float* as_float() const;

    int8_t* as_int8_mut();

    // "(8, 8) int8" — for readable error messages.
    std::string describe() const;

private:
    Dtype dtype_ = Dtype::Int8;
    std::vector<uint32_t> shape_;
    std::vector<uint8_t> raw_;
};

// These throw ModelError if the file doesn't exist, the magic doesn't match, the version doesn't
// agree, or the payload size doesn't match the declared dimensions.
Tensor read_tensor(const std::string& path);
void write_tensor(const std::string& path, const Tensor& tensor);

}  // namespace zaa
