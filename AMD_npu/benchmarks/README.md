# Benchmarks and Performance Evaluation

This directory contains the full benchmarking study, timing analysis, and
experimental observations for the AIE-based tiled GEMM kernel used in vector
retrieval workloads.

All results here are derived from real hardware execution on the AMD Versal AIE.

---

## Contents

### benchmarks.pdf
The primary document summarizing:

• Tile shape selection (32×64×64)
• AIE grid mapping (4×4)
• Streaming hierarchy and DMA scheduling
• Output memory layout discovery
• Row-stride alignment requirements
• Performance results on SIFT-small workload
• Scaling behavior with respect to N
• Debugging path and failure analysis
• Partial reconstruction of hardware output
• Limitations due to undocumented DMA layout

This file should be read to understand the complete engineering process.

---

## Experimental Summary

### Workload
- 128-dimensional SIFT-small embeddings
- 10K database vectors
- 100 query vectors
- GEMM size: 128 × 128 × N, where N = 2048 → 10240

### Metrics Collected
- End-to-end kernel latency via XRT
- Tile load/store overhead
- Streaming pipeline stability
- GFLOP throughput
- Scaling curves with increasing N

---

---

## Notes on Accuracy Evaluation

While the compute kernel is verified to function correctly per tile, accuracy
evaluation (recall@K) could not be performed because:

- The AIE runtime stores outputs in a device-specific DMA tile-streaming format.
- Each tile is column-major and separated by 4 KB alignment gaps.
- The official documentation for reconstructing the full matrix layout is absent.

This does not affect the validity of compute results, but prevents assembling the
final row-major similarity matrix.

---

This directory provides all empirical data needed to assess the performance and
behavior of the AIE GEMM kernel in vector retrieval workloads.
