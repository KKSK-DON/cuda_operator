# CUDA Operator

A collection of hand-written CUDA kernels for common deep learning operators, with progressive optimization strategies (naive → shared memory → warp shuffle → vectorized loads).

## Operators

| Operator | File | Versions | Key Techniques |
|----------|------|----------|----------------|
| Vector Add | `vec_add/vec_add.cu` | 1 | Basic elementwise kernel |
| Softmax | `softmax/softmax.cu` | 5 (v0–v4) | Naive → shared memory reduction → warp shuffle → multi-warp warp shuffle |
| MatMul (GEMM) | `matmul/matmul.cu` | 1 (v0) | Naive global memory, supports alpha/beta scaling |
| RMSNorm | `rmsnorm/rmsnorm.cu` | 2 (v0–v1) | Warp + shared memory reduction → float4 vectorized loads |
| LayerNorm | `layernorm/layernorm.cu` | 3 (v0–v2) | Block reduction → float4 + shared cache → single-pass (Welford-style) float4 |

Each kernel includes a CPU reference implementation and correctness/performance comparison.

## Usage

### Build & Run

```bash
./run.sh <path/to/file.cu>
# Example:
./run.sh softmax/softmax.cu
```

### Profile with Nsight Compute

```bash
./profile.sh <path/to/file.cu>
# Example:
./profile.sh vec_add/vec_add.cu
```

Reports are saved as `.ncu-rep` files in `build/` and can be opened with the Nsight Compute GUI.

## Requirements

- CUDA Toolkit (nvcc)
- MSVC Build Tools (Visual Studio 2022)
- NVIDIA Nsight Compute (for profiling)