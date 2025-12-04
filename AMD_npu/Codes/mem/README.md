Here’s the clean recommendation, **then I’ll generate the full `.md` code for you**:

---

# ✅ Should you write a README for `mm.cc`?

**Yes — but NOT a giant academic explanation.**
Why?

`mm.cc` is a **low-level microkernel**. It is:

* Mostly template-heavy
* Hard to read for non-AIE experts
* Not meant to be understood line-by-line
* Packed with expansions (2×2, 4×2, 4×4) and AIE intrinsics

A README here should:

### ✔️ Explain what the file *is*

### ✔️ Explain what each kernel variant does

### ✔️ Connect to the document (your PDF) at a high level

### ✔️ Explain how to use the kernel (e.g., DIM_M, DIM_K, DIM_N)

### ✔️ Not dump the PDF theory verbatim

This is exactly what Qualcomm/AMD repos do:
**The README describes intent, usage, constraints — not the whole math.**

---

# ✅ What your `mm.cc/README.md` should contain

### **1. Purpose of the file**

Tiled multiply-accumulate microkernels for AIE2 using `aie::mmul`.

### **2. What tile sizes it supports**

Your chosen tile: **32×64×64** globally aligns with:

* r, s, t settings
* 2×2 expansion → int16 kernels
* 4×2 and 4×4 variants → int8, bf16

### **3. Why this file exists**

Explain that the compiler emits calls into these C functions.

### **4. High-level explanation of each kernel**

* `matmul_scalar`
* `matmul_vectorized_2x2_mmul`
* `matmul_vectorized_4x2_mmul`
* `matmul_vectorized_4x4`
* Dispatcher + C wrappers

### **5. How the kernel aligns with your architecture PDF**

Show 4–5 bullets referencing:

* Tile shape rationale
* Accumulator reuse strategy
* Why 2×2 or 4×2 is needed
* Why `row-major` vs `col-major` flags exist
* How expansion increases SIMD utilization

### **6. Important notes**

* Must match DIM_M, DIM_K, DIM_N
* Must match instruction file tiling
* Not row-major output reorder logic — that is handled in host

### **7. What NOT to include**

Do not re-explain the entire architecture or DMA.
Link to your main README instead.

---

# ✅ Now I’ll give you a polished, human-readable, copy-paste-ready

# **`mm.cc/README.md`**

with formatting & Markdown styling.

---

# 📄 **README.md for `mm.cc`**

````markdown
# `mm.cc` — AIE2 Tiled Matrix-Multiplication Microkernels

This file contains the **compute microkernels** used by the Versal AIE2 array
to execute a tiled GEMM of the form:

\[
C[M\times N] = A[M\times K] \times B[K\times N]
\]

It implements multiple vectorized `aie::mmul`-based kernels, supporting
different data types, accumulator sizes, and expansion factors (2×2, 4×2, 4×4).
These kernels form the **core compute stage** of the streaming GEMM pipeline
described in the main project documentation.  
(See the *High-Performance GEMM for Vector Retrieval on Edge NPUs* report for
system diagrams and design rationale.)

---

## 🧩 1. Purpose of This File

`mm.cc` is the **AIE-side compute kernel**. It is loaded into each AIE tile as
part of the compiled `xclbin` and performs:

- Per-tile multiply-accumulate of submatrices  
- Vectorized loading of A and B tiles  
- Accumulator reuse across K-dimension loops  
- Output tile formation in row-major or col-major, depending on flags  

It does **not** handle:
- DMA programming  
- Tile routing  
- Output buffer reordering  
Those tasks are handled on the host (`test.cpp`) and in the MLIR-generated
graph.

---

## 🧱 2. Kernel Variants Implemented

### **2.1 Scalar Reference Kernel**
```cpp
matmul_scalar<T_in, T_out>
````

A simple triple-nested loop implementation.
Used for:

* Debugging
* Small shapes
* Verifying correctness

---

### **2.2 Vectorized 2×2 MMUL Kernel**

```cpp
matmul_vectorized_2x2_mmul
```

Expands compute in both the **M** and **N** directions:

* 2 rows × 2 columns computed per iteration
* Uses `aie::mmul<r,s,t>`
* Good for `int16` inputs and `int32` accumulation
* High SIMD utilization with moderate register pressure

This is the kernel used for the **32×64×64** tile shape described in the design
document.
✔ Matches the compute tile drawn in the architecture diagrams.
✔ Provides strong accumulator reuse.

---

### **2.3 Vectorized 4×2 Kernel**

```cpp
matmul_vectorized_4x2_mmul
```

Expands the `m` dimension four times and the `n` dimension twice.

Used when:

* The datatype is 8-bit (`int8`)
* We want higher accumulator utilization
* We need enough parallel rows to amortize cost of MMUL setup

---

### **2.4 Vectorized 4×4 Kernel**

```cpp
matmul_vectorized_4x4
```

Largest expansion in the file:

* 4 rows × 4 columns per step
* Extremely high MAC throughput
* Used for bf16 and float accumulations

This variant is key for high efficiency on architectures where r=4, s=8, t=4
are natively optimal.

---

## 🔧 3. Template Dispatcher

At the bottom of the file, a set of macros:

```cpp
matmul_vectorized_c_func(...)
matmul_scalar_c_func(...)
```

automatically export C-callable names such as:

```
matmul_i16_i32
matmul_bf16_f32
matmul_i8_i8
```

These functions are what MLIR-AIE and the runtime actually bind to inside the
kernel object.

---

## 📐 4. Tile Size Requirements

Each kernel variant enforces important constraints:

* `m % (2*r) == 0`
* `k % s == 0`
* `n % (2*t) == 0`

For this project, we use a global tile shape of:

* **TM = 32**
* **TN = 64**
* **TK = 64**

which maps cleanly onto the chosen MMUL shapes:

* `(r, s, t) = (4, 4, 4)` for int16
* `(r, s, t) = (4, 8, 8)` for int8
* `(r, s, t) = (4, 8, 4)` for bf16

These shapes were selected based on:

* AIE local memory size
* Vector width
* Accumulator depth
* Streaming reuse efficiency
* Empirical performance (see architecture document)


---

## 🔄 5. Row-Major and Column-Major Options

Two compile-time flags determine memory layout:

* `B_COL_MAJ`: interpret B as column-major
* `C_COL_MAJ`: store outputs in column-major

In this project, we typically use **row-major for C**, because:

* The AIE tile emits column-major within tiles
* The host (`test.cpp`) flattens and reorders them
* The final consumer (retrieval pipeline) expects row-major

See the *output layout section* in the main report for complete explanation.


---

## 🚀 6. How This Kernel Fits Into the Full System

`mm.cc` is **only** the compute step in the hierarchy:

```
Host → DMA / FIFOs → AIE Tile → MMUL Kernels → Output FIFO → DRAM
```

Corresponds to the pipeline described in the architecture report:

* DRAM → Shim → L2 → L1 → MMUL
* Tiles of A stream vertically
* Tiles of B stream horizontally
* Output tiles require reordering on host


---

## 📝 7. Guidance for Modification

If you want to change tile size:

* Adjust `DIM_M`, `DIM_K`, `DIM_N`
* Ensure divisibility constraints still hold
* Adjust MLIR tiling and DMA descriptors
* Ensure host reorder logic is updated

If you want a different data type:

* Use the appropriate MMUL shape
* Add a new macro entry in the dispatcher

---

## 📌 8. Summary

This file implements the **high-performance, vectorized, tile-level compute
kernels** used by the AIE array in this project. It is deeply optimized for:

* Correct tiling
* High SIMD utilization
* Minimal register stall
* Efficient accumulator reuse
* Compatibility with MLIR-AIE lowering

For full architectural context, memory hierarchy diagrams, and the motivation
behind tile selection, refer to the main project documentation.

```

---

# ✅ Done.  
If you want:

- A matching README for `test.cpp`
- A README for `/src/preprocessing`
- A README for the whole `/src` folder  
- Mermaid diagrams inside the README  
- A separate **"Developer Notes" doc** explaining tuning/optimizations  
- A "Kernel Shapes Comparison Table"  

Just tell me — I can generate all of them now.
```
