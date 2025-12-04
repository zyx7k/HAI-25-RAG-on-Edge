# Vector Search on Edge NPUs

High-performance approximate and exact nearest neighbor search implementations optimized for edge Neural Processing Units (NPUs).

## Overview

This repository contains three implementations of vector similarity search:

1. **AMD AIE (Versal NPU)** - Tiled GEMM on AMD AI Engine array
2. **Qualcomm HTP (Snapdragon NPU)** - Brute force search with INT8 quantization
3. **Qualcomm HTP IVF** - Inverted File Index for approximate search
4. **CPU Baseline** - Multi-threaded exact search with SIMD optimization

All implementations target the SIFT dataset (128-dimensional vectors) and are designed for retrieval-augmented generation (RAG) workloads.

## Repository Structure

```
├── AMD_npu/              # AMD Versal AIE implementation
│   ├── Codes/           # MLIR-AIE source and host code
│   └── benchmarks/      # Performance results
├── qidk_bruteforce/     # Qualcomm NPU brute force search
│   ├── android/         # Native C++ code
│   ├── prepare/         # ONNX model generation
│   └── scripts/         # Build and deployment scripts
├── qidk_ivf/           # Qualcomm NPU IVF search
│   ├── android/         # Native C++ with IVF index
│   ├── prepare/         # Index building scripts
│   └── scripts/         # Build and deployment scripts
└── cpu/                # CPU baseline implementation
```

## Implementations

### AMD Versal AIE

**Target**: AMD Versal VCK5000 or similar AIE-enabled FPGAs

**Approach**: Distributes matrix multiplication across a 4×4 AIE tile array using 32×64×64 tile decomposition.

**Key Files**:
- `whole_array.py` - MLIR-AIE graph generation
- `mm.cc` - Vectorized GEMM microkernel
- `test.cpp` - XRT host driver

**Build**: Requires Vitis, MLIR-AIE toolchain, and XRT runtime.

**Performance**: Constrained by undocumented DMA output format. Tile-level compute validated but full end-to-end accuracy evaluation blocked.

### Qualcomm Brute Force

**Target**: Snapdragon 8 Gen 2+ with HTP (Hexagon Tensor Processor)

**Approach**: Single MatMul operation on NPU with INT8 quantization. Batching support for improved throughput.

**Key Files**:
- `create_model.py` - ONNX model generation
- `main.cpp` - Benchmark executable
- `QnnRunner.cpp` - QNN SDK wrapper

**Build**:
```bash
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
export QNN_SDK_ROOT=~/qualcomm/qnn
bash scripts/build.sh siftsmall
bash scripts/deploy.sh siftsmall 10k 5
```

**Performance**: ~5000 QPS (batch=32) on 10K dataset, ~47 GFLOPS

### Qualcomm IVF

**Target**: Same as brute force

**Approach**: Two-stage search using k-means clustering. Coarse search on NPU finds nearest centroids, fine search on CPU (NEON) computes exact distances within selected clusters.

**Key Files**:
- `create_ivf_model.py` - Index building
- `IVFIndex.cpp` - Search implementation
- `main_ivf.cpp` - Benchmark executable

**Build**:
```bash
python3 prepare/create_ivf_model.py sift 1024
bash qnn/convert_centroids.sh sift
bash scripts/build.sh
bash scripts/deploy_ivf.sh sift 32
```

**Performance**: 23-78× speedup vs brute force with 80-91% recall depending on nprobe setting.

### CPU Baseline

**Target**: x86-64 CPU with AVX2

**Approach**: Multi-threaded exact search using OpenBLAS GEMM and AVX2/FMA3 vectorization.

**Build**:
```bash
cd cpu
g++ -o cpu_baseline cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
./cpu_baseline siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_query.fvecs 5 results.txt
```

**Performance**: ~200-500 QPS on SIFT-small (10K), 100% recall.

## Dataset

Uses SIFT vectors from the INRIA corpus:
- **SIFT-small**: 10K base vectors, 100 queries
- **SIFT**: 1M base vectors, 10K queries

Download:
```bash
cd data && mkdir siftsmall && cd siftsmall
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
```

## Requirements

**AMD AIE**:
- Vitis 2023.2+
- MLIR-AIE compiler
- XRT runtime

**Qualcomm**:
- Android NDK r25c
- QNN SDK v2.29.0+
- ADB-connected Snapdragon device

**CPU**:
- OpenBLAS
- OpenMP
- AVX2-capable CPU

## Key Results

| Implementation | Dataset | Throughput | Recall | Speedup |
|---------------|---------|------------|--------|---------|
| AMD AIE | SIFT-small | N/A | N/A | N/A (blocked) |
| QNN Brute (B=1) | SIFT-small | 1042 QPS | 100% | 1× |
| QNN Brute (B=32) | SIFT-small | 5208 QPS | 100% | 1× |
| QNN IVF (np=8) | SIFT-small | ~40K QPS | 82% | 78× |
| QNN IVF (np=32) | SIFT-small | ~24K QPS | 91% | 23× |
| CPU AVX2 | SIFT-small | 200-500 QPS | 100% | baseline |

## Documentation

Each implementation directory contains detailed READMEs:
- `AMD_npu/Codes/README.md` - Host driver details
- `AMD_npu/Codes/mem/README.md` - Microkernel documentation
- `qidk_bruteforce/README.md` - Brute force implementation
- `qidk_ivf/README.md` - IVF implementation
- `cpu/README.md` - CPU baseline

Performance analysis: `AMD_npu/benchmarks/README.md`

## Limitations

**AMD AIE**: Output reordering blocked by undocumented hardware DMA layout. Tile-level compute verified but end-to-end accuracy cannot be measured.

**Qualcomm**: Requires physical Snapdragon device. QNN SDK license required for commercial use.