# Source Directory

This directory contains all source files required to build, configure, and execute
the tiled GEMM kernel on the AMD Versal AI Engine (AIE) as part of the project
“High-Performance GEMM for Vector Retrieval on Edge NPUs (AMD AIE)”.

The components in this folder implement:

• Host-side execution logic (XRT runtime, buffer management)
• AIE kernel generation and graph construction
• Preprocessing for real embedding datasets (SIFT-small)
• Execution pipeline for performance benchmarking

---

## File Overview

### 1. test.cpp
Host program responsible for:
- Allocating input/output buffers using XRT.
- Loading the xclbin and dispatching the AIE kernel.
- Managing row-stride–aligned output buffers.
- Reading SIFT-small vectors and preparing A and B tiles.
- Partially reconstructing output tiles (column-major → row-major within tiles).
- Reporting execution latency.

This is the main entry point for executing the GEMM kernel on hardware.

---

### 2. mm.cc
MLIR-AIE kernel implementation for the 32×64×64 GEMM tile configuration.

Responsibilities:
- Defines the compute kernel that multiplies A (32×64) by B (64×64).
- Uses AIE intrinsics for vector MAC operations.
- Assumes column-major tile format for B and accumulator layout compatible with AIE SIMD width.
- Matches the pre-compiled kernel format used by MLIR-AIE.

---

### 3. array.py
Python script that builds the full AIE computation graph.

Functionality:
- Constructs a configurable grid of compute tiles (default: 4×4).
- Inserts Object FIFOs for streaming A and B tiles.
- Generates ND-DMA descriptors for multi-dimensional tile movement.
- Outputs the final .inst file required by the host program.
- Encodes the hierarchical memory traffic pattern:
  DRAM → SHIM → L2 → L1 → AIE cores → Output FIFO.

---

### 4. preprocessing/
Utility scripts for preparing the SIFT-small dataset.

Tasks:
- Reading `.fvecs` and `.ivecs` formats.
- Normalizing or converting embeddings as needed.
- Arranging embeddings into contiguous buffers matching AIE tile boundaries.
- Generating packed A and B matrices for the GEMM workload.

---

## Usage

Typical workflow inside `src/`:

1. Preprocess SIFT-small vectors:
python3 preprocessing/preprocess_sift.py
2. Generate AIE graph and instructions:
python3 array.py

3. Compile host code:

g++ test.cpp -o run.out -lxrt_coreutil -O3

