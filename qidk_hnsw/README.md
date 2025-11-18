# HNSW Vector Search with Snapdragon NPU Acceleration

High-performance approximate nearest-neighbor search using **HNSW (Hierarchical Navigable Small World)** with **Qualcomm Hexagon NPU (HTP)** acceleration via **QNN SDK**.
Distance computations run on the NPU, while graph traversal remains on CPU.

---

# Overview

This project provides:

* HNSW-based ANN search
* NPU-accelerated distance computation (batched dot products)
* INT8 quantization using QNN SDK
* CPU graph traversal + NPU math hybrid
* QNN context generation from ONNX
* Tested on SIFT datasets (10k and 1M vectors)



---

# Key Features

* Hierarchical graph index (HNSW)
* CPU traversal, NPU math
* QNN HTP backend, no CPU fallback
* Automatic conversion: ONNX → QNN model → QNN context
* INT8 quantization
* Handles 10k–1M vectors
* Optimized NPU batching

---

# Requirements

## Hardware

* Linux x86_64 host machine
* Android device with Snapdragon 8 Gen 2+ (HTP supported)
* ADB connection

## Software

* Python 3.8–3.10
* Android NDK r25c
* Qualcomm QNN SDK v2.29.0+
* ONNX 1.12+
* ADB installed

---

# Environment Setup

Create `.env` or export:

```bash
export ANDROID_NDK_ROOT=$HOME/Android/Sdk/ndk/25.2.9519653/android-ndk-r25c
export QNN_SDK_ROOT=$HOME/qualcomm/qnn
```

---

# Installation

## Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# Dataset Setup

Place files under `data/`.

### Download SIFT Small (10k)

```bash
cd data
mkdir -p siftsmall && cd siftsmall
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
mv siftsmall/* .
rmdir siftsmall
rm siftsmall.tar.gz
cd ../..
```

### Download SIFT (1M)

```bash
cd data
mkdir -p sift && cd sift
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mv sift/* .
rmdir sift
rm sift.tar.gz
cd ../..
```

---

# Directory Structure

```
qidk_hnsw/
├── README.md
├── README_HNSW.md
├── requirements.txt
├── build_hnsw_index.py
├── android/
│   └── app/main/jni/
│       ├── main.cpp
│       ├── hnsw_search.cpp
│       ├── hnsw_search.h
│       ├── QnnRunner.cpp
│       └── QnnRunner.h
├── scripts/
│   ├── build_hnsw.sh
│   ├── deploy_hnsw.sh
│   ├── clean.sh
│   └── verify_setup.sh
├── qnn/
│   ├── convert_to_qnn.sh
│   ├── calibration/
│   └── qnn_artifacts/
└── data/
    ├── siftsmall/
    └── sift/
```

---

# Build Pipeline

```bash
bash scripts/build_hnsw.sh <dataset> <M> <ef_construction>
```

Parameters:

* `dataset`: `siftsmall` or `sift`
* `M`: graph connectivity (8–64, recommended 16)
* `ef_construction`: 100–800 (recommended 200)

Build performs:

1. Create ONNX matmul model
2. Convert ONNX → QNN
3. Generate quantized context
4. Build HNSW index
5. Build Android executable

---

# Deployment and Execution

```bash
bash scripts/deploy_hnsw.sh <dataset> <M> <top_k> <ef_search>
```

Parameters:

* `dataset`: must match build
* `M`: must match build
* `top_k`: neighbors to return
* `ef_search`: accuracy vs. speed (50–500)

Outputs saved under:

`results/<dataset>_M<M>_ef<ef_search>/`

---

# HNSW Parameter Guide

## M (graph connectivity)

| M  | Memory    | Speed  | Accuracy  | Use Case           |
| -- | --------- | ------ | --------- | ------------------ |
| 8  | Low       | Fast   | Medium    | Low-memory systems |
| 16 | Medium    | Medium | Good      | Recommended        |
| 32 | High      | Slower | Very Good | High recall        |
| 64 | Very High | Slow   | Excellent | Maximum accuracy   |

## ef_construction

| ef_c | Build Time | Quality   |
| ---- | ---------- | --------- |
| 100  | Fast       | Basic     |
| 200  | Medium     | Good      |
| 400  | Slow       | Very Good |
| 800  | Very Slow  | Excellent |

## ef_search

| ef_s | Speed     | Recall |
| ---- | --------- | ------ |
| 50   | Fast      | ~0.95  |
| 100  | Medium    | ~0.97  |
| 200  | Slow      | ~0.99  |
| 500  | Very Slow | ~0.995 |

---

# Performance

## SIFT Small (10k)

* Build: 5–10 seconds
* Search: 1–5 ms/query
* Memory: ~10–20 MB

## SIFT (1M)

* Build: 10–20 minutes
* Search: 10–50 ms/query
* Memory: 500MB–1GB

Metrics include:

* Average latency
* P50/P95/P99
* CPU vs NPU time breakdown
* Recall

---

# Troubleshooting

### ndk-build not found

Set:

```bash
export ANDROID_NDK_ROOT=/path/to/ndk
```

### QNN SDK not found

Set:

```bash
export QNN_SDK_ROOT=/path/to/qnn
```

### No device detected

```bash
adb devices
adb kill-server
adb start-server
```

### Context generation failed

* Device may not support HTP
* Check logs:

```bash
adb logcat | grep -i qnn
```

### Low recall

* Increase `ef_search`
* Increase `M` & `ef_construction`

---

# License

For research and educational use.
SIFT dataset © INRIA.
Qualcomm QNN SDK subject to Qualcomm license.

---