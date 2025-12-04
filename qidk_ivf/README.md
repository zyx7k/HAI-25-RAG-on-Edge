# IVF Vector Search on Qualcomm NPU

NPU-accelerated Inverted File Index (IVF) implementation for approximate nearest neighbor search on Qualcomm Snapdragon devices.

## Requirements

### Software
- Android NDK 25.x or later
- Qualcomm QNN SDK 2.x
- Python 3.8+
- ADB (Android Debug Bridge)

### Hardware
- Qualcomm Snapdragon device with HTP (Hexagon Tensor Processor)
- Tested on Snapdragon 8 Gen 2

## Setup

### 1. Environment Variables

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
export QNN_SDK_ROOT=/path/to/qnn-sdk
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place SIFT dataset files in the `data/` directory:
```
data/
├── sift/
│   ├── sift_base.fvecs
│   ├── sift_query.fvecs
│   └── sift_groundtruth.ivecs
└── siftsmall/
    ├── siftsmall_base.fvecs
    ├── siftsmall_query.fvecs
    └── siftsmall_groundtruth.ivecs
```

## Building

### Build IVF Index

```bash
# For siftsmall (10K vectors)
python prepare/create_ivf_model.py siftsmall

# For sift (1M vectors)
python prepare/create_ivf_model.py sift 1024
```

### Convert Model to QNN Format

```bash
./qnn/convert_centroids.sh sift 32
```

### Build C++ Executable

```bash
./scripts/build.sh
```

## Deployment

Deploy to Android device:

```bash
./scripts/deploy_ivf.sh sift 32
```

## Running

On the Android device:

```bash
adb shell
cd /data/local/tmp/ivf_search
export LD_LIBRARY_PATH=.
./qidk_ivf ./ivf_sift sift_query.fvecs results ./libQnnHtp.so 10 32 sift_groundtruth.ivecs
```

### Command Arguments

| Argument | Description |
|----------|-------------|
| `index_dir` | Path to IVF index directory |
| `queries.fvecs` | Query vectors file |
| `results_dir` | Output directory for results |
| `backend.so` | QNN backend library (libQnnHtp.so) |
| `top_k` | Number of results to return |
| `nprobe` | Number of clusters to search |
| `groundtruth.ivecs` | Ground truth file (optional) |

## Benchmarking

Run Python benchmark:

```bash
python prepare/benchmark_ivf.py sift 100
```

## Configuration

### nprobe Parameter

| nprobe | Recall@10 | Speedup | Use Case |
|--------|-----------|---------|----------|
| 8 | ~82% | 78x | Maximum speed |
| 16 | ~88% | 40x | Balanced |
| 32 | ~91% | 23x | High accuracy |

### Index Files

| File | Description |
|------|-------------|
| `centroids.onnx` | Centroid model for coarse search |
| `centroids.bin` | QNN context binary |
| `ivf_config.json` | Index configuration |
| `cluster_offsets.npy` | Cluster boundaries |
| `cluster_indices.npy` | Vector IDs per cluster |
| `vectors.npy` | Database vectors |

## Directory Structure

```
qidk_ivf/
├── prepare/              # Python scripts for index building
├── android/app/main/jni/ # C++ implementation
├── qnn/                  # QNN conversion scripts
├── scripts/              # Build and deployment scripts
├── models/               # Generated model files
└── data/                 # Dataset files
```
