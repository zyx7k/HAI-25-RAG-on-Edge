# HAI-25-RAG-on-Edge

High-performance k-NN vector search implementations optimized for edge devices, with CPU baseline, NPU-accelerated brute-force, and HNSW approximate search.

## Overview

This project implements and compares three approaches for k-Nearest Neighbor (k-NN) vector search on SIFT datasets:

1. **CPU Baseline** - Exact brute-force search with AVX2/OpenBLAS optimizations
2. **QIDK Brute-Force** - Exact search accelerated by Qualcomm NPU via QNN SDK
3. **QIDK HNSW** - Approximate search using HNSW graph with NPU acceleration

All implementations target edge deployment on Qualcomm Snapdragon devices with Hexagon NPU.

## Project Structure

```
HAI-25-RAG-on-Edge/
├── cpu/                    # CPU baseline implementation
│   ├── cpu_baseline.cpp    # Optimized C++ baseline
│   └── README.md           # Build and run instructions
│
├── qidk_bruteforce/        # NPU-accelerated brute-force
│   ├── android/            # Android NDK build
│   ├── qnn/                # QNN model conversion
│   └── README.md           # Setup and deployment guide
│
├── qidk_hnsw/              # NPU-accelerated HNSW
│   ├── android/            # Android NDK build
│   ├── qnn/                # QNN model conversion
│   ├── src/                # HNSW index builder
│   └── README.md           # Setup and deployment guide
│
└── benchmarks/             # Comparison tools and results
    ├── compare_accuracy.py # Accuracy comparison script
    ├── results/            # Stored benchmark results
    └── README.md           # Benchmarking guide
```

## Quick Start

### CPU Baseline

```bash
cd cpu
g++ -o cpu_baseline cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
./cpu_baseline
```

### NPU Implementations

**Prerequisites:**
- Android device with Snapdragon 8 Gen 2+
- Qualcomm QNN SDK v2.29.0+
- Android NDK r25c
- ADB connection

**Setup:**
```bash
# Set environment variables
export ANDROID_NDK_ROOT=/path/to/ndk
export QNN_SDK_ROOT=/path/to/qnn

# For brute-force
cd qidk_bruteforce
bash scripts/build.sh siftsmall
bash scripts/deploy.sh siftsmall

# For HNSW
cd qidk_hnsw
bash scripts/build.sh siftsmall
bash scripts/deploy.sh siftsmall
```

### Benchmark Comparison

```bash
cd benchmarks
python compare_accuracy.py
```

## Performance Summary

### SIFT-small (10K vectors, 100 queries, k=5)

| Implementation | Latency (ms/query) | Throughput (q/s) | Accuracy |
|----------------|-------------------|------------------|----------|
| CPU Baseline   | 2-5              | 200-500          | 100%     |
| QIDK Brute     | ~1-3             | 300-1000         | >99.5%   |
| QIDK HNSW      | ~0.5-2           | 500-2000         | 95-99%   |

*Performance varies by device, CPU/NPU model, and configuration.*

## Requirements

### CPU Implementation
- C++11 compiler (GCC 5.0+)
- OpenMP
- OpenBLAS
- AVX2 + FMA3 support

### NPU Implementations
- Linux x86_64 host
- Android device with Qualcomm Snapdragon NPU
- Qualcomm QNN SDK
- Android NDK r25c
- Python 3.8-3.10
- ADB

## Documentation

- **[CPU Baseline](cpu/README.md)** - CPU implementation details
- **[QIDK Brute-Force](qidk_bruteforce/README.md)** - NPU brute-force guide
- **[QIDK HNSW](qidk_hnsw/README.md)** - NPU HNSW guide
- **[Benchmarks](benchmarks/README.md)** - Comparison methodology
- **[Git Configuration](GIT_CONFIGURATION.md)** - Version control setup

## Key Features

### CPU Baseline
- ✅ Exact k-NN search (100% recall)
- ✅ Multi-threading (OpenMP)
- ✅ SIMD optimization (AVX2/FMA3)
- ✅ Optimized BLAS (GEMM)
- ✅ Comprehensive metrics (latency, percentiles)

### QIDK Implementations
- ✅ Qualcomm Hexagon NPU acceleration
- ✅ INT8 quantization via QNN SDK
- ✅ Automatic ONNX → QNN conversion
- ✅ Batch processing optimization
- ✅ On-device execution

### HNSW-specific
- ✅ Approximate nearest neighbor search
- ✅ Graph-based index structure
- ✅ Tunable recall/speed tradeoff
- ✅ Scales to 1M+ vectors

## Datasets

Uses SIFT datasets from [INRIA TEXMEX](http://corpus-texmex.irisa.fr/):

- **SIFT-small**: 10K base vectors, 100 queries (128-dim)
- **SIFT**: 1M base vectors, 10K queries (128-dim)

Dataset files are automatically downloaded during setup or can be manually placed in respective `data/` directories.

## Accuracy Metrics

All implementations are benchmarked against the CPU baseline (ground truth):

- **Top-1 Accuracy**: Exact match of nearest neighbor
- **Top-K Accuracy**: Overlap in top-K results
- **Recall@K**: Standard k-NN recall metric

Run `benchmarks/compare_accuracy.py` for detailed accuracy reports.

## Use Cases

- **Semantic Search**: Document/image retrieval
- **RAG Systems**: Retrieval-augmented generation
- **Recommendation**: Similar item finding
- **Anomaly Detection**: Outlier identification
- **Edge AI**: On-device vector search

## Contributing

Contributions are welcome! Areas of interest:
- Additional optimization techniques
- New acceleration targets (GPU, other NPUs)
- Alternative index structures (IVF, ScaNN)
- Extended benchmarking

## License

This project is part of academic coursework (Hardware for AI - Semester 7).

## Acknowledgments

- **INRIA TEXMEX** for SIFT datasets
- **Qualcomm** for QNN SDK and NPU platform
- **OpenBLAS** project for optimized BLAS
- **HNSW** algorithm by Malkov & Yashunin

## Contact

For questions or issues, please open a GitHub issue in this repository.

---

**Quick Links:**
- [CPU Setup](cpu/README.md)
- [NPU Brute-Force](qidk_bruteforce/README.md)
- [NPU HNSW](qidk_hnsw/README.md)
- [Benchmarking](benchmarks/README.md)
