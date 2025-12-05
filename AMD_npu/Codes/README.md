# **README — test.cpp (Host Driver for AIE Tiled GEMM Execution)**

### *C++ Host Orchestration for Tiled GEMM on AMD Versal AIE*

---

## **1. Purpose of `test.cpp`**

`test.cpp` is the **host-side runtime controller** for executing the tiled GEMM kernel on the AMD Versal AI Engine (AIE).
It is responsible for:

* Loading matrices **A**, **B**, and instruction streams
* Allocating and managing XRT buffer objects (BOs)
* Launching the AIE compute kernel
* Handling DMA synchronization
* Reordering the raw tile-streamed output into row-major form
* (Optionally) verifying correctness against a CPU reference
* Measuring performance (latency, GFLOPs)

**this file acts as the bridge between the MLIR-AIE-generated compute graph and the actual hardware execution flow.**

---

## **2. High-Level Execution Flow**

The flow in `test.cpp` mirrors the architecture described in the report:
DRAM → SHIM → L2 → L1 → AIE compute → DMA → Host buffer 

### **Step-by-step:**

1. **Parse command-line arguments**
   Using `cxxopts`, the host accepts parameters such as:

   * `M, N, K` sizes
   * number of iterations
   * warmup count
   * paths to `A.bin`, `B.bin`, `instr.bin`
   * verbosity levels
   * verification toggles

2. **Load input matrices A and B**
   Binary files (`A.bin`, `B.bin`) are read into host vectors.

3. **Allocate XRT buffers**
   Buffers for:

   * A, B input matrices
   * Output matrix C
   * Instruction stream
   * Temporary buffers
   * Trace buffer (optional)

4. **Copy host data into device buffers**
   Uses memory mapping + `memcpy` to write into AIE-accessible memory.

5. **Load and register the XCLBIN**
   The AIE kernel is fetched using its name in the XCLBIN container.

6. **Execute the GEMM kernel**
   The kernel is dispatched with all required BOs and opcode = `3`.

7. **Read back the raw output**
   The output is initially in **tile-major, column-major-per-tile form**,
   with **4 KB-aligned gaps** inserted by the DMA hardware.
   This behavior matches the behavior documented in the report
   (AIE streams tiles, not full rows) .

8. **Reorder output into row-major format**
   A custom routine flattens:

   * each tile (32×64)
   * from column-major
   * to global row-major (M × N matrix)


10. **Performance reporting**
    Prints:

* average, min, max latency
* achieved GFLOPs
* iteration statistics

---

## **3. Key Technical Highlights (What this Program Actually Solves)**

### **Handles the undocumented AIE DMA output format**

AIE does **not** return row-major matrices.
It streams tiles *in tile order*, with **column-major storage inside each tile**, and padding between them.
The reorder logic in `test.cpp` is the host fix for this — exactly what the debugging uncovered. 

### **Correctly chooses tile sizes (32×64×64)**

Matches the AIE kernel generated in MLIR-AIE and fits local memory constraints.

### **Works around 4 KB alignment constraints**

the program respects this by using the correct buffer allocation size.

### **Integrates with real SIFTsmall data**

Reads `A.bin` and `B.bin` produced from the embedding preprocessing stage.

### **Provides robust runtime verification**

Includes:

* full L2 matmul
* stochastic sample-based verification
* tolerance handling for integer accumulation

### **Collects consistent benchmarking results**

Used in the PDF performance report.

---

## **4. Output Reordering Logic**

The most critical part of `test.cpp` is reconstructing the flattened matrix from AIE’s tile-streamed output.

### What the hardware gives you:

```
Tile0 (col-major) | 4KB pad | Tile1 | 4KB pad | ...
```

### What you need:

```
Row-major matrix: C[M × N]
```

### How the code fixes it:

```cpp
int tile_index = tile_m * tiles_per_row + tile_n;
int tile_offset = tile_index * TM * TN;

for (int im = 0; im < TM; im++) {
    for (int in = 0; in < TN; in++) {
        int src = tile_offset + in * TM + im;           // col-major inside tile
        int dst = (tile_m * TM + im) * N + (tile_n * TN + in);
        CRowMajor[dst] = CVec[src];
    }
}
```

This implements exactly the reverse-engineering described in the report, where tile-major output needed host post-processing. 

---

## **5. Verification Strategy**

`test.cpp` supports two modes:

### **Full verification**

Used when:

```
M * N * K <= 1GB
```

### **Stochastic sampling**

Used when matrix sizes are very large:

* 1000 random (i, j) positions
* Computes dot products in software
* Checks absolute + relative error tolerances

Both are defined in `matmul_common`.

---

## **6. Performance Measurement**

For each iteration:

* Record start/end timestamps
* Compute GFLOPs using:

```
MACs = 2 * M * K * N
GFLOPs = MACs / time_us / 1000
```

Warmup iterations are excluded.

---

## **7. When to Use This Program**

Use this host program when:

* You want to run GEMM workloads on AMD Versal AIE
* You have generated an `.xclbin` and an instruction stream
* You have input matrices (e.g., SIFT vectors) in binary row-major format
* You need performance measurements or verification
* You need to debug AIE tile streaming behavior

---

## **8. Usage Example**

### Build:

```
g++ test.cpp -std=c++17 -lxrt_coreutil -O3 -o run.out
```

### Run:

```
./run.out --xclbin aie_gemm.xclbin --instr inst.bin --M 128 --K 128 --N 10240 --verify
```

### Input files expected:

```
A.bin
B.bin
inst.bin
```

### Output:

```
C_output.bin
timing statistics
optional trace file
```

---
