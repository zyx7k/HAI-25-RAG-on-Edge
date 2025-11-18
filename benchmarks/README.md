# Benchmarks and Accuracy Comparison

This directory contains tools and results for comparing the performance and accuracy of different k-NN search implementations (CPU baseline, NPU-accelerated, QIDK, HNSW, etc.).

## Overview

The benchmarking framework provides:
- **Accuracy Comparison**: Top-K accuracy metrics between different implementations
- **Result Analysis**: Systematic storage and comparison of search results
- **Performance Metrics**: Aggregated statistics across multiple runs

## Directory Structure

```
benchmarks/
├── README.md                      # This file
├── compare_accuracy.py            # Accuracy comparison tool
├── results/                       # Stored results from various implementations
│   ├── siftsmall_results.txt      # CPU baseline (SIFT-small)
│   ├── siftsmall_results_qidk.txt # QIDK results (SIFT-small)
│   ├── sift_results.txt           # CPU baseline (SIFT)
│   ├── sift_results_qidk.txt      # QIDK results (SIFT)
│   ├── cpu.txt                    # Legacy CPU results
│   └── npu.txt                    # Legacy NPU results
```

## Tools

### Accuracy Comparison Script

`compare_accuracy.py` compares the accuracy of different k-NN implementations by computing Top-K accuracy metrics.

#### Features
- Computes Top-1, Top-3, and Top-5 accuracy
- Supports multiple datasets (SIFT-small, SIFT)
- Automatic result parsing and validation
- Comprehensive summary statistics

#### Usage

```bash
cd benchmarks
python compare_accuracy.py
```

#### Output Example
```
======================================================================
k-NN ACCURACY COMPARISON: CPU vs QIDK
======================================================================

======================================================================
SIFT-SMALL DATASET COMPARISON
======================================================================

Reference file: results/siftsmall_results.txt
Test file: results/siftsmall_results_qidk.txt
Number of queries: 100

Top-1 Accuracy: 98.00%
Top-3 Accuracy: 99.33%
Top-5 Accuracy: 99.60%

======================================================================
SIFT DATASET COMPARISON
======================================================================

Reference file: results/sift_results.txt
Test file: results/sift_results_qidk.txt
Number of queries: 10000

Top-1 Accuracy: 97.50%
Top-3 Accuracy: 98.83%
Top-5 Accuracy: 99.20%

======================================================================
SUMMARY
======================================================================

SIFT-small:
  Top-1: 98.00%
  Top-3: 99.33%
  Top-5: 99.60%

SIFT:
  Top-1: 97.50%
  Top-3: 98.83%
  Top-5: 99.20%
```

## Result File Format

All result files follow a standardized format for easy parsing and comparison:

```
Query 0: (3752, 76269.5) (2176, 77121.2) (9345, 78234.8) (5621, 79123.4) (8234, 80234.5)
Query 1: (4521, 65432.1) (8765, 66543.2) (1234, 67654.3) (9876, 68765.4) (3456, 69876.5)
...
```

Each line contains:
- **Query ID**: Sequential query identifier (0-indexed)
- **Results**: Space-separated tuples of `(document_id, distance)`
- **Ordering**: Results sorted by distance in ascending order

## Adding New Results

To add results from a new implementation:

1. Generate results in the standard format
2. Save the file in `results/` directory with a descriptive name:
   - `<dataset>_results_<implementation>.txt`
   - Example: `sift_results_hnsw.txt`

3. Update `compare_accuracy.py` to include your comparison:
```python
datasets = [
    ("SIFT-small", "siftsmall_results.txt", "siftsmall_results_<your_impl>.txt"),
    ("SIFT", "sift_results.txt", "sift_results_<your_impl>.txt")
]
```

## Accuracy Metrics

### Top-K Accuracy

Top-K accuracy measures the overlap between the reference (CPU baseline) and test implementation:

```
Accuracy@K = (1/Q) × Σ |Reference_K ∩ Test_K| / K
```

Where:
- `Q` = Total number of queries
- `Reference_K` = Top-K results from reference implementation
- `Test_K` = Top-K results from test implementation
- `|A ∩ B|` = Size of set intersection

### Interpretation

- **100%**: Perfect match with reference (exact search)
- **>99%**: Excellent accuracy, negligible quality loss
- **95-99%**: Good accuracy, acceptable for most applications
- **90-95%**: Fair accuracy, may require tuning
- **<90%**: Poor accuracy, indicates issues with implementation

## Performance Comparison

For performance metrics (latency, throughput), refer to the output of each implementation:
- **CPU Baseline**: See `cpu/` folder and run `./cpu_baseline`
- **QIDK/NPU**: Check implementation-specific output logs
- **HNSW**: See `qidk_hnsw/` folder results

## Best Practices

### Running Comparisons

1. **Generate CPU baseline first**:
   ```bash
   cd ../cpu
   ./cpu_baseline
   ```

2. **Run alternative implementation**:
   ```bash
   cd ../qidk_bruteforce  # or qidk_hnsw
   # Follow implementation-specific instructions
   ```

3. **Copy results to benchmarks**:
   ```bash
   cp <implementation>/output/*.txt ../benchmarks/results/
   ```

4. **Run comparison**:
   ```bash
   cd ../benchmarks
   python compare_accuracy.py
   ```

### Result Management

- Use consistent naming conventions
- Include timestamp or version in filename for multiple runs
- Archive old results before generating new ones
- Document any changes to parameters (k, dataset, etc.)

## Datasets

### SIFT-small
- **Base vectors**: 10,000
- **Query vectors**: 100
- **Dimension**: 128
- **Use case**: Quick validation and development

### SIFT
- **Base vectors**: 1,000,000
- **Query vectors**: 10,000
- **Dimension**: 128
- **Use case**: Full-scale evaluation

## Troubleshooting

### File Not Found Errors
```
Error: File 'results/siftsmall_results.txt' not found
```
**Solution**: Ensure you've run the CPU baseline and copied results to `benchmarks/results/`

### Parsing Errors
```
Error parsing file: ...
```
**Solution**: Verify result file follows the correct format (see [Result File Format](#result-file-format))

### Inconsistent Query Counts
```
Warning: Reference has 100 queries but test has 99
```
**Solution**: Ensure both implementations processed the same query file

## Expected Accuracy Ranges

Based on implementation characteristics:

| Implementation | Expected Top-5 Accuracy | Notes |
|----------------|-------------------------|-------|
| CPU Baseline | 100.00% | Exact search (reference) |
| QIDK Brute-force | >99.5% | Quantization effects |
| HNSW | 95-99% | Approximate search |
| NPU Optimized | >99% | Architecture-specific |

## Contributing

When adding new comparison tools:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include error handling
4. Update this README with usage instructions

## License

This benchmarking framework is part of the HAI-25-RAG-on-Edge project.

## References

- [SIFT Dataset](http://corpus-texmex.irisa.fr/)
- [Recall@K Metric](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
