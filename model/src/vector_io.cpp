#include "vector_io.hpp"

#include <cstdio>
#include <cstring>

#include "common.hpp"

namespace zaa {
namespace {

constexpr char kMagic[4] = {'Z', 'A', 'A', 'V'};

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void write_u32_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

}  // namespace

std::size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::Int8:
            return 1;
        case Dtype::Int32:
            return 4;
        case Dtype::Float32:
            return 4;
    }
    throw ModelError("unknown dtype");
}

const char* dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::Int8:
            return "int8";
        case Dtype::Int32:
            return "int32";
        case Dtype::Float32:
            return "float32";
    }
    return "?";
}

Tensor::Tensor(Dtype dtype, std::vector<uint32_t> shape) : dtype_(dtype), shape_(std::move(shape)) {
    if (shape_.empty() || shape_.size() > kMaxDims) {
        throw ModelError("a tensor must have between 1 and 4 dimensions");
    }
    raw_.assign(numel() * dtype_size(dtype_), 0);
}

std::size_t Tensor::numel() const {
    std::size_t n = 1;
    for (uint32_t d : shape_) {
        n *= d;
    }
    return n;
}

const int8_t* Tensor::as_int8() const {
    if (dtype_ != Dtype::Int8) {
        throw ModelError("asked for int8 but the tensor is " + std::string(dtype_name(dtype_)));
    }
    return reinterpret_cast<const int8_t*>(raw_.data());
}

int8_t* Tensor::as_int8_mut() {
    if (dtype_ != Dtype::Int8) {
        throw ModelError("asked for int8 but the tensor is " + std::string(dtype_name(dtype_)));
    }
    return reinterpret_cast<int8_t*>(raw_.data());
}

const int32_t* Tensor::as_int32() const {
    if (dtype_ != Dtype::Int32) {
        throw ModelError("asked for int32 but the tensor is " + std::string(dtype_name(dtype_)));
    }
    return reinterpret_cast<const int32_t*>(raw_.data());
}

const float* Tensor::as_float() const {
    if (dtype_ != Dtype::Float32) {
        throw ModelError("asked for float32 but the tensor is " + std::string(dtype_name(dtype_)));
    }
    return reinterpret_cast<const float*>(raw_.data());
}

std::string Tensor::describe() const {
    std::string s = "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape_[i]);
    }
    s += ") ";
    s += dtype_name(dtype_);
    return s;
}

Tensor read_tensor(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw ModelError("could not open " + path +
                         " (did you run 'python3 reference/gen_vectors.py --smoke'?)");
    }

    uint8_t header[kVectorHeaderSize];
    if (std::fread(header, 1, kVectorHeaderSize, f) != kVectorHeaderSize) {
        std::fclose(f);
        throw ModelError(path + ": truncated file, not even enough bytes for the header");
    }

    if (std::memcmp(header, kMagic, 4) != 0) {
        std::fclose(f);
        throw ModelError(path + ": invalid magic, this does not look like a ZAAV file");
    }

    const uint32_t version = read_u32_le(header + 4);
    if (version != kVectorFormatVersion) {
        std::fclose(f);
        throw ModelError(path + ": format version " + std::to_string(version) + ", this binary reads version " +
                         std::to_string(kVectorFormatVersion) + " — regenerate the vectors");
    }

    const uint32_t dtype_code = read_u32_le(header + 8);
    if (dtype_code > static_cast<uint32_t>(Dtype::Float32)) {
        std::fclose(f);
        throw ModelError(path + ": unknown dtype code " + std::to_string(dtype_code));
    }
    const Dtype dtype = static_cast<Dtype>(dtype_code);

    const uint32_t ndim = read_u32_le(header + 12);
    if (ndim == 0 || ndim > kMaxDims) {
        std::fclose(f);
        throw ModelError(path + ": ndim out of range: " + std::to_string(ndim));
    }

    std::vector<uint32_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_u32_le(header + 16 + 4 * i);
    }

    Tensor t(dtype, shape);
    const std::size_t expected = t.size_bytes();
    const std::size_t got = std::fread(t.raw().data(), 1, expected, f);

    // It must have read exactly what was expected, with nothing left over afterwards.
    uint8_t extra;
    const bool has_extra = std::fread(&extra, 1, 1, f) == 1;
    std::fclose(f);

    if (got != expected || has_extra) {
        throw ModelError(path + ": payload size does not match the dimensions " +
                         t.describe() + " (expected " + std::to_string(expected) + " bytes)");
    }

    return t;
}

void write_tensor(const std::string& path, const Tensor& tensor) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw ModelError("could not write " + path);
    }

    uint8_t header[kVectorHeaderSize];
    std::memset(header, 0, sizeof(header));
    std::memcpy(header, kMagic, 4);
    write_u32_le(header + 4, kVectorFormatVersion);
    write_u32_le(header + 8, static_cast<uint32_t>(tensor.dtype()));
    write_u32_le(header + 12, static_cast<uint32_t>(tensor.shape().size()));
    for (std::size_t i = 0; i < kMaxDims; ++i) {
        const uint32_t dim = i < tensor.shape().size() ? tensor.shape()[i] : 1u;
        write_u32_le(header + 16 + 4 * i, dim);
    }

    const bool ok = std::fwrite(header, 1, sizeof(header), f) == sizeof(header) &&
                    std::fwrite(tensor.raw().data(), 1, tensor.size_bytes(), f) == tensor.size_bytes();
    std::fclose(f);

    if (!ok) {
        throw ModelError("incomplete write to " + path);
    }
}

}  // namespace zaa
