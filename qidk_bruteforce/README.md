# QNN Vector Search

High-performance vector similarity search using Qualcomm's Neural Processing Unit (NPU) via the QNN SDK.

## Overview

This project demonstrates NPU-accelerated vector search on Snapdragon devices. It uses INT8 quantization and batch processing to achieve high throughput for nearest-neighbor queries on large vector databases.

**Key Features:**
- NPU acceleration via Qualcomm AI Engine Direct (QNN SDK)
- INT8 quantization with NEON-optimized preprocessing
- Batched inference for improved throughput
- Support for SIFT datasets (10K and 1M vectors)

## Requirements

### Hardware
- Qualcomm QIDK or Snapdragon-powered Android device
- USB debugging enabled

### Software
- Linux host (Ubuntu 20.04+ recommended)
- Android NDK r25c
- Qualcomm QNN SDK v2.29.0
- Python 3.8-3.10
- ADB

### Environment Setup

```bash
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
export QNN_SDK_ROOT=~/qualcomm/qnn
```

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
cd data && mkdir -p siftsmall && cd siftsmall
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz && mv siftsmall/* . && rm -rf siftsmall siftsmall.tar.gz
cd ../..
```

### 3. Build and Run

```bash
# Build for siftsmall dataset
bash scripts/build.sh siftsmall

# Deploy and run benchmark
bash scripts/deploy.sh siftsmall 10k 5
```

### 4. Run Full Benchmark Suite

```bash
# Run with default batch sizes (1, 8, 16, 32, 64)
bash scripts/run_all.sh siftsmall

# Or specify custom batch sizes
bash scripts/run_all.sh sift 1,8,32,64
```

## Scripts

| Script | Description |
|--------|-------------|
| `clean.sh` | Remove build artifacts |
| `build.sh` | Build ONNX model, convert to QNN, compile native code |
| `deploy.sh` | Deploy to device and run inference |
| `run_all.sh` | Run benchmarks across multiple batch sizes |

### Usage Examples

```bash
# Clean build artifacts
bash scripts/clean.sh

# Build for 1M dataset with batch size 32
bash scripts/build.sh sift 1M_b32

# Deploy and retrieve top-10 results
bash scripts/deploy.sh sift 1M_b32 10

# Run full benchmark suite
bash scripts/run_all.sh siftsmall 1,8,16,32,64
```

## Project Structure

```
├── android/app/main/jni/   # Native C++ code
│   ├── main.cpp            # Benchmark executable
│   ├── QnnRunner.cpp       # QNN SDK wrapper
│   └── QnnRunner.h
├── data/                   # Vector datasets (SIFT)
├── models/                 # ONNX models
├── prepare/
│   └── create_model.py     # Model generation
├── qnn/                    # QNN conversion scripts
├── results/                # Benchmark outputs
└── scripts/                # Build and run scripts
```

## Results

Results are saved to `results/<dataset>_<size>/`:
- `results.txt` - Top-K results for each query
- `metrics.txt` - Performance metrics (throughput, latency, GFLOPS)

### Sample Output

```
Dataset Information:
  Number of queries: 100
  Number of documents: 10000
  Batch size: 32

Overall Performance:
  Throughput: 5208 queries/sec
  Avg GFLOPS: 47.5
```

## Troubleshooting

**Device not detected:**
```bash
adb devices  # Verify connection
adb kill-server && adb start-server  # Restart ADB
```

**Context binary generation failed:**
- Ensure device has sufficient storage
- Check device compatibility with HTP backend

**Build errors:**
- Verify `ANDROID_NDK_ROOT` and `QNN_SDK_ROOT` are set correctly
- Ensure NDK version is r25c

## License

MIT License
