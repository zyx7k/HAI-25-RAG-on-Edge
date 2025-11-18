# CPU Baseline for k-NN Search

High-Performance Exact k-NN (SIFT-Small 10K)

A high-performance, **CPU-only**, **exact** brute-force k-Nearest Neighbor (k-NN) search implementation.
Used as a baseline for evaluating accelerated versions (NPU/HNSW/etc.).
Computes all pairwise Euclidean distances between query vectors and base vectors.

---

# Key Features

* **Exact Search** – 100% accuracy
* **Multi-Threading (OpenMP)** – parallelized norm, distance, and top-k loops
* **SIMD Vectorization** – AVX2 + FMA3 intrinsics
* **Optimized GEMM** – OpenBLAS `cblas_sgemm` for Q·Bᵀ
* **Cache-friendly O(N·k) Top-K** – faster than heap when k is small
* **Detailed Performance Metrics** – mean, std, percentiles

---

# Requirements

## Hardware

* CPU with **AVX2** and **FMA3** (Intel Haswell+, AMD Excavator+)
* ≥ 4GB RAM

## Software

* C++11 compiler (GCC/Clang/MSVC)
* OpenMP
* OpenBLAS
* SIFT-small dataset (10K)

---

# Dataset Setup

Download SIFT-small:

```bash
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
```

Expected structure:

```
siftsmall/
├── siftsmall_base.fvecs
├── siftsmall_query.fvecs
└── siftsmall_groundtruth.ivecs
```

---

# Build Instructions

## Linux/macOS

```bash
g++ -o cpu_baseline cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
```

## Windows (MinGW)

```bash
g++ -o cpu_baseline.exe cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
```

## Compiler Flags Explained

* `-O3` – aggressive optimizations
* `-fopenmp` – multi-threading
* `-lopenblas` – optimized GEMM
* `-mavx2` and `-mfma` – vectorization and fused multiply-add

---

# How to Run

```bash
./cpu_baseline <base_file> <query_file> <k> <output_file>
```

Example:

```bash
./cpu_baseline siftsmall/siftsmall_base.fvecs \
               siftsmall/siftsmall_query.fvecs \
               5 \
               results.txt
```

Arguments:

| Argument    | Description                    |
| ----------- | ------------------------------ |
| base_file   | Path to base `.fvecs` vectors  |
| query_file  | Path to query `.fvecs` vectors |
| k           | Number of nearest neighbors    |
| output_file | Output file for results        |

---

# Pipeline Description

### 1. Read Vectors (`read_fvecs`)

* Reads binary `.fvecs` format
* Loads vectors into contiguous memory (cache-friendly)

### 2. Norm Computation (`compute_norms`)

* Computes squared norms `||v||²`
* SIMD accelerated (AVX2 + FMA3)
* Parallelized with OpenMP

### 3. Dot Product Computation (`compute_dot_products`)

Uses GEMM:

[
C = Q \cdot B^T
]

* Q: (num_queries × dim)
* B: (num_base × dim)
* OpenBLAS `cblas_sgemm` handles blocking + SIMD

### 4. Distance Conversion (`convert_to_distances`)

Applies:

[
d(q,b) = ||q||^2 + ||b||^2 - 2(q\cdot b)
]

* Vectorized with AVX2
* Parallel over queries

### 5. Top-K Selection (`find_knn`)

* Linear O(N·k) scan (faster than heaps for small k)
* Per-query parallelism with OpenMP

---

# Performance Expectations

### SIFT-small (10K base vectors)

* **Latency**: ~2–5 ms/query
* **Throughput**: 200–500 qps
* **Accuracy**: exact

### SIFT (1M base vectors)

* **Latency**: 100–300 ms/query
* **Throughput**: 3–10 qps
* **Accuracy**: exact

---

# Output Format

Example `results.txt`:

```
Query 0: (3752, 76269.5) (2176, 77121.2) (9345, 78234.8) ...
Query 1: (4521, 65432.1) (8765, 66543.2) (1234, 67654.3) ...
```

Each tuple = (document_index, distance)

---

# Troubleshooting

### Missing OpenBLAS (`cblas.h` not found)

```bash
sudo apt-get install libopenblas-dev
```

### Missing OpenMP (`omp.h` not found)

```bash
sudo apt-get install libomp-dev
```

### CPU does not support AVX2

Check:

```bash
lscpu | grep avx2
```

Remove flags:

```bash
-mavx2 -mfma
```

### Slow performance

* Ensure OpenMP is active
* Set CPU governor to performance:

```bash
sudo cpupower frequency-set -g performance
```

---

# Modifying Parameters

Change default k:

```cpp
int k = 5;
```

Run only one dataset by editing `main()`.

---

# License

Research and educational use.
SIFT dataset © INRIA.

---

# References

* SIFT Dataset – [http://corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/)
* OpenBLAS – [https://www.openblas.net/](https://www.openblas.net/)
* Intel Intrinsics Guide – [https://www.intel.com/content/www/us/en/docs/intrinsics-guide/](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
