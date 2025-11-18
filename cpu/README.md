# CPU Baseline for k-NN Search# High-Performance k-NN Baseline (SIFT 10K)



High-performance CPU-only implementation of exact k-Nearest Neighbor (k-NN) search using brute-force distance computation. This baseline establishes the performance and accuracy reference for comparing accelerated implementations (NPU, HNSW, etc.).This project is a high-performance, **brute-force** C++ baseline for k-Nearest Neighbor (k-NN) search. It finds the *exact* nearest neighbors for a set of query vectors from a base set by computing all pairwise Euclidean distances.



## Overview---



This implementation performs exact k-NN search by computing all pairwise Euclidean distances between query vectors and a database of document vectors. It serves as the ground truth baseline for accuracy comparisons and performance benchmarking.## Key Optimizations



### Key Features* **Multi-Threading (OpenMP):** All major computational loops (norm calculation, distance conversion, Top-K search) are parallelized to use all available CPU cores.

* **Vectorization (SIMD):** CPU-bound "helper" functions (norm and distance calculation) use **AVX2 and FMA3** instructions to process 8 floating-point numbers in a single instruction.

- **Exact Search**: Guarantees 100% accuracy by exhaustive distance computation* **BLAS (GEMM):** The core dot-product calculation ($Q \cdot B^T$) is offloaded to an optimized BLAS library (OpenBLAS) via the `cblas_sgemm` function. This is the fastest possible way to perform matrix multiplication on a CPU.

- **Multi-Threading**: OpenMP parallelization across all available CPU cores* **Optimized Top-K Search:** Uses a cache-friendly $O(N \cdot k)$ linear scan instead of a heap-based $O(N \log k)$ method, which is faster for small $k$.

- **SIMD Optimization**: AVX2 and FMA3 instructions for vectorized computation

- **BLAS Integration**: OpenBLAS (GEMM) for optimized matrix multiplication---

- **Comprehensive Metrics**: Detailed latency statistics (mean, std dev, percentiles)

## How to Run

## Requirements

### 1. Get the Data

### Hardware

- CPU with AVX2 and FMA3 support (Intel Haswell or later, AMD Excavator or later)This program is designed for the SIFT 10K dataset.

- Minimum 4GB RAM (for SIFT dataset)

```bash

### Software# 1. Download the SIFT 10K dataset (5.1MB)

- C++ compiler with C++11 support (GCC 5.0+, Clang 3.9+, or MSVC 2015+)curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz

- OpenMP support

- OpenBLAS library# 2. Extract the files

tar -zxvf siftsmall.tar.gz

### Installation on Ubuntu/Debian# This will create a 'siftsmall/' directory containing the .fvecs files

```bash```

sudo apt-get update

sudo apt-get install build-essential libomp-dev libopenblas-dev---

```

### 2. Compile the Program

### Installation on macOS

```bashThis code **requires** a CPU with **AVX2 and FMA3** support. You must pass the correct flags to the compiler to enable these instruction sets.

brew install gcc libomp openblas

``````bash

# Compile the program

### Installation on Windowsg++ -o baseline baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma

- Install MinGW-w64 or Visual Studio```

- Install OpenBLAS from [here](https://github.com/xianyi/OpenBLAS/releases)

**Explanation of flags:**

## Building

* `-o baseline`: Creates an executable file named `baseline`.

### Linux/macOS* `-O3`: Enables high-level compiler optimizations.

```bash* `-fopenmp`: Enables OpenMP for multi-threading.

cd cpu* `-lopenblas`: Links the OpenBLAS library (for `cblas_sgemm`).

g++ -o cpu_baseline cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma* `-mavx2`: Tells the compiler to generate AVX2 instructions.

```* `-mfma`: Tells the compiler to generate FMA (Fused Multiply-Add) instructions.



### Compiler Flags Explained---

- `-O3`: Maximum optimization level

- `-fopenmp`: Enable OpenMP multi-threading### 3. Run the Search

- `-lopenblas`: Link OpenBLAS library

- `-mavx2`: Enable AVX2 SIMD instructionsThe program is run from the command line, providing the data files and $k$ as arguments.

- `-mfma`: Enable FMA3 (Fused Multiply-Add) instructions

```bash

### Windows (MinGW)# Run the baseline

```bash./baseline siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_query.fvecs 5 knn_results.txt

g++ -o cpu_baseline.exe cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma```

```

**Command-Line Arguments:**

## Running the Baseline```

./baseline [base_file] [query_file] [k] [output_file]

The program expects the SIFT dataset files to be organized in the following structure:```

```

cpu/* **`[base_file]`**: `sift/sift_base.fvecs` (The 10,000 document vectors)

├── cpu_baseline.cpp* **`[query_file]`**: `sift/sift_query.fvecs` (The 100 query vectors)

├── cpu_baseline          # or cpu_baseline.exe on Windows* **`[k]`**: The number of neighbors to find

├── sift/* **`[output_file]`**: `results.txt` (The file to save results to)

│   ├── sift_base.fvecs

│   ├── sift_query.fvecs---

│   └── ...

└── siftsmall/## Code Deep Dive

    ├── siftsmall_base.fvecs

    ├── siftsmall_query.fvecsThe program executes in a 5-stage pipeline, managed by the `main()` function.

    └── ...

```---



### Execute the Benchmark### 1. File I/O (`read_fvecs`)

```bash

./cpu_baseline* **What:** Reads the `.fvecs` binary format.

```* **How:** The format is a repeating sequence of `[int32 dimension, (float * dimension)]`.

* **Optimization:** The code reads all vectors into a single, contiguous `std::vector<float>`.  

The program will automatically:  This flat memory layout is crucial for fast, cache-friendly access in later steps.

1. Run on the SIFT-small dataset (10K base vectors, 100 queries)

2. Run on the SIFT dataset (1M base vectors, 10K queries)---

3. Generate result files: `siftsmall_results.txt` and `sift_results.txt`

4. Display comprehensive performance metrics### 2. Norm Computation (`compute_norms`)



### Expected Output* **What:** Pre-calculates the squared Euclidean norm ($||v||^2$) for every vector in the query and base sets.

```* **Why:** The full distance formula is  

=== CPU Baseline for k-NN Search ===  $$d^2(q, b) = ||q||^2 + ||b||^2 - 2(q \cdot b)$$  

Using 8 OpenMP threads  This step pre-computes the first two parts.

SIMD: AVX2 + FMA3 enabled* **Optimizations:**

====================================  * `#pragma omp parallel for`: The main loop is split across all CPU cores.

  * `compute_norm_avx2`: Uses AVX2 intrinsics. Loads 8 floats into a `__m256` register and uses `_mm256_fmadd_ps` (Fused Multiply-Add) to compute $v^2$ and accumulate the sum in one instruction.

========================================

Processing dataset: SIFT-small---

========================================

### 3. Dot Product (GEMM) (`compute_dot_products`)

Loading base file: siftsmall/siftsmall_base.fvecs

Loading query file: siftsmall/siftsmall_query.fvecs* **What:** The most computationally expensive step — computes dot products for all 100 queries against 10,000 base vectors.

Pre-computing norms...* **Why:** This calculates the $(q \cdot b)$ term for all pairs.

* **Optimization:**  

=== CPU RAG Performance Metrics ===  This is formulated as a **General Matrix-Matrix Multiplication (GEMM):**



Dataset Information:  * Let $Q$ be the ($100 \times 128$) query matrix.

  Number of queries: 100  * Let $B$ be the ($10,000 \times 128$) base matrix.

  Number of documents: 10000  * Compute **$C = Q \cdot B^T$**, where $B^T$ is ($128 \times 10,000$).

  Dimension: 128  * The result $C$ is ($100 \times 10,000$), where $C_{ij}$ is the dot product of query $i$ and base $j$.

  Top-K: 5

* This is done with a single call to `cblas_sgemm` (OpenBLAS), which handles all threading, cache-blocking, and vectorization internally.

Overall Performance:

  Total execution time: 0.245 s---

  Throughput: 408.16 queries/sec

### 4. Distance Conversion (`convert_to_distances`)

Distance Computation:

  Average latency: 2.35 ms/query* **What:** Assembles final distances via  

  P50 latency: 2.31 ms  $$D_{ij} = ||q_i||^2 + ||b_j||^2 - 2 \cdot C_{ij}$$

  P95 latency: 2.48 ms* **Optimizations:**

  P99 latency: 2.52 ms  * `#pragma omp parallel for`: Outer loop (over queries) parallelized.

  ...  * `convert_to_distances_avx2`: Uses AVX2 to apply the formula to 8 distances at a time — maximizing arithmetic throughput and minimizing memory bandwidth.

```

---

## Algorithm Details

### 5. Top-K Selection (`find_knn`)

### Pipeline Stages

* **What:** For each query (100 total), finds indices of the $k$ smallest distances among 10,000.

1. **Data Loading**: Read `.fvecs` binary format into memory* **Optimizations:**

2. **Norm Pre-computation**: Calculate ||v||² for all vectors using AVX2  * `#pragma omp parallel for`: Each query handled by a separate thread.

3. **Distance Computation**:   * `select_topk`: Implements an efficient $O(N \cdot k)$ scan (no heap).  

   - Matrix multiplication Q·Bᵀ via OpenBLAS GEMM    Maintains a small array of size $k$, updates max index, and linearly scans through all distances.  

   - Distance formula: d²(q,b) = ||q||² + ||b||² - 2(q·b)    For small $k$, this outperforms heap-based $O(N \log k)$ methods and is more cache-friendly.

4. **Top-K Selection**: O(N·k) linear scan for k smallest distances

5. **Results Output**: Write (index, distance) pairs to file---


### Optimization Strategies

**SIMD Vectorization (AVX2)**
- Processes 8 floats per instruction
- Used in norm computation and distance conversion
- ~4-8x speedup over scalar code

**Multi-threading (OpenMP)**
- Parallel norm computation
- Parallel per-query processing
- Scales linearly with core count

**Optimized BLAS (GEMM)**
- Cache-blocked matrix multiplication
- Assembly-optimized kernels
- Typically 10-100x faster than naive loops

**Efficient Top-K**
- O(N·k) linear scan vs O(N log k) heap
- Better cache locality for small k
- Faster for k << N

## Performance Expectations

### SIFT-small (10K documents, 100 queries)
- **Latency**: ~2-5 ms per query (8-core CPU)
- **Throughput**: ~200-500 queries/sec
- **Accuracy**: 100% (exact search)

### SIFT (1M documents, 10K queries)
- **Latency**: ~100-300 ms per query (8-core CPU)
- **Throughput**: ~3-10 queries/sec
- **Accuracy**: 100% (exact search)

*Note: Performance varies based on CPU model, core count, and memory bandwidth.*

## Output Format

Results are saved in text files with the following format:
```
Query 0: (3752, 76269.5) (2176, 77121.2) (9345, 78234.8) ...
Query 1: (4521, 65432.1) (8765, 66543.2) (1234, 67654.3) ...
...
```

Each line contains:
- Query ID
- List of (document_index, distance) pairs in ascending order by distance
- Top-K pairs (default K=5)

## Troubleshooting

### Compilation Errors

**Error: "cblas.h not found"**
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev
```

**Error: "omp.h not found"**
```bash
# Install OpenMP
sudo apt-get install libomp-dev
```

**Error: AVX2/FMA instructions not supported**
- Verify CPU support: `lscpu | grep avx2`
- Remove `-mavx2 -mfma` flags (performance will degrade)

### Runtime Errors

**Error: "Cannot open file siftsmall_base.fvecs"**
- Ensure dataset files are in the correct directories
- Check file paths match the expected structure

**Slow Performance**
- Verify OpenMP is enabled: program should show `Using N OpenMP threads`
- Check CPU governor: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Set to performance mode: `sudo cpupower frequency-set -g performance`

## Modifying Parameters

To change k (number of neighbors), edit the `k` variable in `main()`:
```cpp
int k = 5;  // Change this value
```

To run on a single dataset, comment out unwanted `run_benchmark()` calls in `main()`.

## License

This code is part of the HAI-25-RAG-on-Edge project.

## References

- SIFT Dataset: [INRIA TEXMEX](http://corpus-texmex.irisa.fr/)
- OpenBLAS: [https://www.openblas.net/](https://www.openblas.net/)
- AVX2 Intrinsics: [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
