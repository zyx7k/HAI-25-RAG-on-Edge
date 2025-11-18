# Benchmark Results Summary

This document provides a systematic overview of all stored benchmark results.

## Result Files

### SIFT-small Dataset Results

| File | Implementation | Date Generated | Queries | Description |
|------|---------------|----------------|---------|-------------|
| `siftsmall_results.txt` | CPU Baseline | - | 100 | Exact k-NN search on CPU using AVX2+OpenBLAS |
| `siftsmall_results_qidk.txt` | QIDK/NPU | - | 100 | NPU-accelerated k-NN search via Qualcomm QIDK |

### SIFT Dataset Results

| File | Implementation | Date Generated | Queries | Description |
|------|---------------|----------------|---------|-------------|
| `sift_results.txt` | CPU Baseline | - | 10000 | Exact k-NN search on CPU using AVX2+OpenBLAS |
| `sift_results_qidk.txt` | QIDK/NPU | - | 10000 | NPU-accelerated k-NN search via Qualcomm QIDK |

### Legacy Results

| File | Implementation | Description |
|------|---------------|-------------|
| `cpu.txt` | CPU Legacy | Previous CPU baseline results |
| `npu.txt` | NPU Legacy | Previous NPU results |

## Dataset Specifications

### SIFT-small
- **Base Vectors**: 10,000
- **Query Vectors**: 100
- **Dimensions**: 128
- **Format**: .fvecs (binary float vectors)
- **K (neighbors)**: 5

### SIFT
- **Base Vectors**: 1,000,000
- **Query Vectors**: 10,000
- **Dimensions**: 128
- **Format**: .fvecs (binary float vectors)
- **K (neighbors)**: 5

## Accuracy Results

Run `compare_accuracy.py` to generate current accuracy metrics comparing CPU baseline with other implementations.

### Expected Metrics

#### SIFT-small (100 queries)
- Top-1 Accuracy: ~98%+
- Top-3 Accuracy: ~99%+
- Top-5 Accuracy: ~99.5%+

#### SIFT (10,000 queries)
- Top-1 Accuracy: ~97%+
- Top-3 Accuracy: ~98%+
- Top-5 Accuracy: ~99%+

## Performance Metrics

### CPU Baseline Performance

#### SIFT-small
- **Average Latency**: 2-5 ms per query
- **Throughput**: 200-500 queries/sec
- **Distance Computation**: ~60-70% of total time
- **Top-K Selection**: ~30-40% of total time

#### SIFT
- **Average Latency**: 100-300 ms per query
- **Throughput**: 3-10 queries/sec
- **Distance Computation**: ~60-70% of total time
- **Top-K Selection**: ~30-40% of total time

### NPU/QIDK Performance

Performance metrics vary based on:
- Device model (Qualcomm chip generation)
- Quantization settings
- Optimization level
- Power mode

Refer to QIDK implementation folders for specific metrics.

## Adding New Results

When adding new benchmark results:

1. **File Naming Convention**: `<dataset>_results_<implementation>.txt`
   - Example: `sift_results_hnsw.txt`

2. **Update This Summary**: Add entry to the appropriate table above

3. **Document Parameters**:
   - Implementation details
   - Hardware specifications
   - Optimization flags
   - Date generated

4. **Run Accuracy Comparison**:
   ```bash
   python compare_accuracy.py
   ```

## Reproducibility

To reproduce these results:

### CPU Baseline
```bash
cd ../cpu
g++ -o cpu_baseline cpu_baseline.cpp -O3 -fopenmp -lopenblas -mavx2 -mfma
./cpu_baseline
cp siftsmall_results.txt ../benchmarks/results/
cp sift_results.txt ../benchmarks/results/
```

### QIDK/NPU
Refer to the respective implementation directories:
- `qidk_bruteforce/` for brute-force NPU implementation
- `qidk_hnsw/` for HNSW NPU implementation

## Notes

- All results use K=5 (top-5 nearest neighbors)
- CPU baseline uses exact search (100% recall)
- Distance metric: Euclidean (L2)
- Result format: `Query <id>: (<doc_id>, <distance>) ...`

## Last Updated

This summary should be updated whenever new results are added to the `results/` directory.

---

For detailed comparison and analysis, run:
```bash
python compare_accuracy.py
```
