# High-Performance k-NN Baseline (SIFT 10K)

This project is a high-performance, **brute-force** C++ baseline for k-Nearest Neighbor (k-NN) search. It finds the *exact* nearest neighbors for a set of query vectors from a base set by computing all pairwise Euclidean distances.

---

## Key Optimizations

* **Multi-Threading (OpenMP):** All major computational loops (norm calculation, distance conversion, Top-K search) are parallelized to use all available CPU cores.
* **Vectorization (SIMD):** CPU-bound "helper" functions (norm and distance calculation) use **AVX2 and FMA3** instructions to process 8 floating-point numbers in a single instruction.
* **BLAS (GEMM):** The core dot-product calculation ($Q \cdot B^T$) is offloaded to an optimized BLAS library (OpenBLAS) via the `cblas_sgemm` function. This is the fastest possible way to perform matrix multiplication on a CPU.
* **Optimized Top-K Search:** Uses a cache-friendly $O(N \cdot k)$ linear scan instead of a heap-based $O(N \log k)$ method, which is faster for small $k$.

---

## How to Run

### 1. Get the Data

This program is designed for the SIFT 10K dataset.

```bash
# 1. Download the SIFT 10K dataset (5.1MB)
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz

# 2. Extract the files
tar -zxvf siftsmall.tar.gz
# This will create a 'siftsmall/' directory containing the .fvecs files
```

---

### 2. Compile the Program

This code **requires** a CPU with **AVX2 and FMA3** support. You must pass the correct flags to the compiler to enable these instruction sets.

```bash
# Compile the program
g++ -o baseline baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
```

**Explanation of flags:**

* `-o baseline`: Creates an executable file named `baseline`.
* `-O3`: Enables high-level compiler optimizations.
* `-fopenmp`: Enables OpenMP for multi-threading.
* `-lopenblas`: Links the OpenBLAS library (for `cblas_sgemm`).
* `-mavx2`: Tells the compiler to generate AVX2 instructions.
* `-mfma`: Tells the compiler to generate FMA (Fused Multiply-Add) instructions.

---

### 3. Run the Search

The program is run from the command line, providing the data files and $k$ as arguments.

```bash
# Run the baseline
./baseline siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_query.fvecs 5 knn_results.txt
```

**Command-Line Arguments:**
```
./baseline [base_file] [query_file] [k] [output_file]
```

* **`[base_file]`**: `sift/sift_base.fvecs` (The 10,000 document vectors)
* **`[query_file]`**: `sift/sift_query.fvecs` (The 100 query vectors)
* **`[k]`**: The number of neighbors to find
* **`[output_file]`**: `results.txt` (The file to save results to)

---

## Code Deep Dive

The program executes in a 5-stage pipeline, managed by the `main()` function.

---

### 1. File I/O (`read_fvecs`)

* **What:** Reads the `.fvecs` binary format.
* **How:** The format is a repeating sequence of `[int32 dimension, (float * dimension)]`.
* **Optimization:** The code reads all vectors into a single, contiguous `std::vector<float>`.  
  This flat memory layout is crucial for fast, cache-friendly access in later steps.

---

### 2. Norm Computation (`compute_norms`)

* **What:** Pre-calculates the squared Euclidean norm ($||v||^2$) for every vector in the query and base sets.
* **Why:** The full distance formula is  
  $$d^2(q, b) = ||q||^2 + ||b||^2 - 2(q \cdot b)$$  
  This step pre-computes the first two parts.
* **Optimizations:**
  * `#pragma omp parallel for`: The main loop is split across all CPU cores.
  * `compute_norm_avx2`: Uses AVX2 intrinsics. Loads 8 floats into a `__m256` register and uses `_mm256_fmadd_ps` (Fused Multiply-Add) to compute $v^2$ and accumulate the sum in one instruction.

---

### 3. Dot Product (GEMM) (`compute_dot_products`)

* **What:** The most computationally expensive step — computes dot products for all 100 queries against 10,000 base vectors.
* **Why:** This calculates the $(q \cdot b)$ term for all pairs.
* **Optimization:**  
  This is formulated as a **General Matrix-Matrix Multiplication (GEMM):**

  * Let $Q$ be the ($100 \times 128$) query matrix.
  * Let $B$ be the ($10,000 \times 128$) base matrix.
  * Compute **$C = Q \cdot B^T$**, where $B^T$ is ($128 \times 10,000$).
  * The result $C$ is ($100 \times 10,000$), where $C_{ij}$ is the dot product of query $i$ and base $j$.

* This is done with a single call to `cblas_sgemm` (OpenBLAS), which handles all threading, cache-blocking, and vectorization internally.

---

### 4. Distance Conversion (`convert_to_distances`)

* **What:** Assembles final distances via  
  $$D_{ij} = ||q_i||^2 + ||b_j||^2 - 2 \cdot C_{ij}$$
* **Optimizations:**
  * `#pragma omp parallel for`: Outer loop (over queries) parallelized.
  * `convert_to_distances_avx2`: Uses AVX2 to apply the formula to 8 distances at a time — maximizing arithmetic throughput and minimizing memory bandwidth.

---

### 5. Top-K Selection (`find_knn`)

* **What:** For each query (100 total), finds indices of the $k$ smallest distances among 10,000.
* **Optimizations:**
  * `#pragma omp parallel for`: Each query handled by a separate thread.
  * `select_topk`: Implements an efficient $O(N \cdot k)$ scan (no heap).  
    Maintains a small array of size $k$, updates max index, and linearly scans through all distances.  
    For small $k$, this outperforms heap-based $O(N \log k)$ methods and is more cache-friendly.

---
