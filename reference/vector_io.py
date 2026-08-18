"""Reading and writing the golden vector binary format.

This is the contract between the Python reference and the C++ simulator.
The spec lives in docs/DECISIONS.md under D-001, and the mirror implementation is in
model/include/vector_io.hpp. Any change goes on both sides at once.

Layout:
    offset  size    field
    0       4       magic "ZAAV"
    4       4       version  (uint32 LE)
    8       4       dtype    (uint32 LE: 0=int8, 1=int32, 2=float32)
    12      4       ndim     (uint32 LE, <= 4)
    16      16      dims[4]  (uint32 LE x4, unused ones are 1)
    32      ...     raw C-contiguous little-endian data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"ZAAV"
VERSION = 1
HEADER_SIZE = 32
MAX_DIMS = 4

# These codes must match the Dtype enum in vector_io.hpp exactly.
DTYPE_TO_CODE = {
    np.dtype(np.int8): 0,
    np.dtype(np.int32): 1,
    np.dtype(np.float32): 2,
}
CODE_TO_DTYPE = {code: dt for dt, code in DTYPE_TO_CODE.items()}


def write_tensor(path: str | Path, array: np.ndarray) -> None:
    """Writes `array` to `path` in the ZAAV format.

    The array must be int8, int32, or float32, and have at most 4 dimensions.
    """
    array = np.ascontiguousarray(array)

    if array.dtype not in DTYPE_TO_CODE:
        raise ValueError(
            f"unsupported dtype: {array.dtype}. Supported: int8, int32, float32."
        )
    if array.ndim > MAX_DIMS:
        raise ValueError(f"at most {MAX_DIMS} dimensions, got {array.ndim}")

    dims = list(array.shape) + [1] * (MAX_DIMS - array.ndim)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, DTYPE_TO_CODE[array.dtype], array.ndim))
        f.write(struct.pack("<4I", *dims))
        f.write(array.tobytes(order="C"))


def read_tensor(path: str | Path) -> np.ndarray:
    """Reads a ZAAV-format tensor and returns it as an np.ndarray."""
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError(f"{path}: truncated file, not even enough bytes for the header")

        magic = header[:4]
        if magic != MAGIC:
            raise ValueError(f"{path}: invalid magic {magic!r}, expected {MAGIC!r}")

        version, dtype_code, ndim = struct.unpack("<III", header[4:16])
        if version != VERSION:
            raise ValueError(
                f"{path}: version {version}, this code writes/reads version {VERSION}. "
                "Regenerate the vectors with reference/gen_vectors.py."
            )
        if dtype_code not in CODE_TO_DTYPE:
            raise ValueError(f"{path}: unknown dtype code {dtype_code}")
        if ndim == 0 or ndim > MAX_DIMS:
            raise ValueError(f"{path}: ndim out of range: {ndim}")

        dims = struct.unpack("<4I", header[16:32])
        shape = tuple(dims[:ndim])
        dtype = CODE_TO_DTYPE[dtype_code]

        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        payload = f.read()
        if len(payload) != expected_bytes:
            raise ValueError(
                f"{path}: expected {expected_bytes} bytes of data, read {len(payload)}"
            )

    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def describe(path: str | Path) -> str:
    """Returns a readable description of a vector file, for debugging."""
    arr = read_tensor(path)
    return f"{Path(path).name}: shape={arr.shape} dtype={arr.dtype}"
