# Vector Search on Qualcomm NPU

> High-performance vector similarity search using Qualcomm's AI Engine Direct (QNN SDK) on Snapdragon devices

[![Platform](https://img.shields.io/badge/Platform-Qualcomm%20QIDK-green)]()
[![Android](https://img.shields.io/badge/Android-21%2B-blue)]()

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project demonstrates **high-performance vector similarity search** using the Qualcomm Hexagon Neural Processing Unit (NPU) on Snapdragon-powered devices. It leverages:

- **QNN SDK (Qualcomm AI Engine Direct)** for NPU acceleration
- **ONNX** models for neural network representation
- **SIFT datasets** (10k and 1M vectors) for benchmarking
- **Cosine similarity** for vector search

The system achieves significant performance improvements by offloading matrix multiplication operations to the Hexagon DSP/NPU, making it ideal for edge AI applications requiring fast vector retrieval.

---

## Features

- **NPU-Accelerated Search**: Leverages Qualcomm Hexagon for fast vector operations
- **Multi-Dataset Support**: Works with both siftsmall (10k) and sift (1M) datasets
- **Configurable TOP-K**: Retrieve any number of nearest neighbors
- **Comprehensive Metrics**: Latency, throughput, percentiles (p50/p95/p99)

---

## System Requirements

### Hardware
- **Device**: Qualcomm QIDK or Snapdragon-powered Android device
- **Chipset**: Snapdragon 8 Gen 1/2/3 (or compatible with Hexagon DSP)
- **Memory**: 4GB+ RAM recommended (8GB+ for 1M dataset)
- **Storage**: 2GB+ free space

### Software
- **Host OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Android**: API Level 21+ (Android 5.0+)
- **Python**: 3.8 - 3.10
- **ADB**: Android Debug Bridge installed and configured

---

## 📦 Prerequisites

### 1. Android NDK

**Required Version**: `r25c (25.2.9519653)`

**Installation**:
```bash
# Download Android NDK r25c
cd ~/Downloads
wget https://dl.google.com/android/repository/android-ndk-r25c-linux.zip

# Extract to Android SDK location
mkdir -p ~/Android/Sdk/ndk
unzip android-ndk-r25c-linux.zip -d ~/Android/Sdk/ndk/

# Set environment variable
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
```

**Add to your `~/.bashrc` or `~/.zshrc`**:
```bash
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
export PATH=$ANDROID_NDK_ROOT:$PATH
```

### 2. Qualcomm QNN SDK (AI Engine Direct)

**Required Version**: `v2.29.0.241129`

**Installation**:
1. Download from [Qualcomm Developer Network](https://qdn.qualcomm.com/)
   - You'll need to create a Qualcomm account
   - Navigate to AI Engine Direct SDK downloads
   - Download QNN SDK v2.29.0.241129

2. Extract the SDK:
```bash
# Create directory
mkdir -p ~/qualcomm

# Extract (adjust filename as needed)
unzip qairt-2.29.0.241129.zip -d ~/qualcomm/

# Rename/link to standard name
mv ~/qualcomm/qairt ~/qualcomm/qnn
# OR create a symlink
ln -s ~/qualcomm/qairt ~/qualcomm/qnn

# Set environment variable
export QNN_SDK_ROOT=~/qualcomm/qnn
```

**Add to your `~/.bashrc` or `~/.zshrc`**:
```bash
export QNN_SDK_ROOT=~/qualcomm/qnn
export PATH=$QNN_SDK_ROOT/bin:$PATH
```

### 3. Python Environment

```bash
# Install Python 3.8-3.10 if not already installed
sudo apt update
sudo apt install python3 python3-venv python3-pip

# Verify Python version
python3 --version  # Should be 3.8, 3.9, or 3.10
```

### 4. Android Debug Bridge (ADB)

```bash
# Ubuntu/Debian
sudo apt install adb

# macOS
brew install android-platform-tools

# Verify installation
adb --version
```

### 5. Device Setup

1. **Enable Developer Options** on your Android device:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times

2. **Enable USB Debugging**:
   - Go to Settings → Developer Options
   - Enable "USB Debugging"

3. **Connect Device**:
   ```bash
   # Connect via USB
   adb devices
   
   # You should see your device listed
   # If prompted on device, allow USB debugging
   ```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/qidk-rag-demo.git
cd qidk-rag-demo
```

### Step 2: Verify Environment Variables

```bash
# Check that environment variables are set
echo $ANDROID_NDK_ROOT
echo $QNN_SDK_ROOT

# If not set, export them now
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
export QNN_SDK_ROOT=~/qualcomm/qnn
```

### Step 3: Install Python Dependencies

The project will automatically create a virtual environment and install dependencies, but you can also do it manually:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Required Python packages** (automatically installed):
- `onnx==1.13.1` - ONNX model format (required for QNN compatibility)
- `protobuf==3.20.3` - Protocol buffers (required by ONNX)
- `numpy<1.24` - NumPy (version constraint for ONNX compatibility)
- `pyyaml==6.0` - YAML configuration
- `packaging==21.3` - Package version utilities
- `requests` - HTTP library
- `pandas` - Data analysis (optional)

---

## 📥 Dataset Setup

### Option 1: Download SIFT Small Dataset (Recommended for Testing)

**Size**: ~18 MB compressed  
**Vectors**: 10,000 base vectors, 100 query vectors  
**Dimensions**: 128

```bash
cd data
mkdir -p siftsmall && cd siftsmall

# Download
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz

# Extract
tar -xzf siftsmall.tar.gz

# Organize files
mv siftsmall/* .
rmdir siftsmall
rm siftsmall.tar.gz

cd ../..
```

### Option 2: Download SIFT Dataset (For Benchmarking)

**Size**: ~550 MB compressed  
**Vectors**: 1,000,000 base vectors, 10,000 query vectors  
**Dimensions**: 128

```bash
cd data
mkdir -p sift && cd sift

# Download (this may take a while)
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

# Extract
tar -xzf sift.tar.gz

# Organize files
mv sift/* .
rmdir sift
rm sift.tar.gz

cd ../..
```

### Expected Directory Structure

After downloading, your `data/` directory should look like this:

```
data/
├── siftsmall/
│   ├── siftsmall_base.fvecs       # 10,000 base vectors
│   ├── siftsmall_query.fvecs      # 100 query vectors
│   ├── siftsmall_groundtruth.ivecs
│   └── siftsmall_learn.fvecs
└── sift/
    ├── sift_base.fvecs            # 1,000,000 base vectors
    ├── sift_query.fvecs           # 10,000 query vectors
    ├── sift_groundtruth.ivecs
    └── sift_learn.fvecs
```

---

## Usage

### Quick Start: Run All Datasets

The easiest way to get started is to run the automated script that processes both datasets:

```bash
bash scripts/run_all_datasets.sh
```

This will:
1. Build models for both siftsmall and sift datasets
2. Convert them to QNN format
3. Deploy and run on your connected Android device
4. Save results to separate directories

### Run Individual Datasets

#### SIFT Small (10k vectors)

```bash
# Build the project
bash scripts/build.sh siftsmall

# Deploy and run (default: TOP_K=5)
bash scripts/deploy.sh siftsmall

# Or specify custom TOP_K
bash scripts/deploy.sh siftsmall 10k 10  # Returns top-10 results
```

#### SIFT (1M vectors)

```bash
# Build the project
bash scripts/build.sh sift

# Deploy and run
bash scripts/deploy.sh sift

# Or specify custom TOP_K
bash scripts/deploy.sh sift 1M 20  # Returns top-20 results
```

### Command-Line Options

#### build.sh

```bash
bash scripts/build.sh [dataset_name] [model_size_suffix]
```

- `dataset_name`: `siftsmall` or `sift` (default: `siftsmall`)
- `model_size_suffix`: `10k`, `1M`, etc. (auto-detected)

#### deploy.sh

```bash
bash scripts/deploy.sh [dataset_name] [model_size_suffix] [top_k]
```

- `dataset_name`: `siftsmall` or `sift` (default: `siftsmall`)
- `model_size_suffix`: `10k`, `1M`, etc. (auto-detected)
- `top_k`: Number of results to return (default: `5`)

**Examples**:
```bash
# Run siftsmall with top-5 results
bash scripts/deploy.sh siftsmall 10k 5

# Run sift with top-10 results
bash scripts/deploy.sh sift 1M 10

# Use defaults (siftsmall, auto-detect size, top-5)
bash scripts/deploy.sh
```

### Advanced: Manual Steps

If you want more control, you can run each step manually:

```bash
# 1. Create ONNX model
python3 prepare/create_model.py siftsmall

# 2. Convert to QNN format
bash qnn/convert_to_qnn.sh siftsmall

# 3. Build C++ executable
cd android/app/main/jni
$ANDROID_NDK_ROOT/ndk-build QNN_SDK_ROOT=$QNN_SDK_ROOT
cd ../../../..

# 4. Deploy and run
bash scripts/deploy.sh siftsmall 10k 5
```

---

## 📂 Project Structure

```
qidk-rag-demo/
├── android/                    # Android native code
│   ├── app/main/
│   │   ├── jni/
│   │   │   ├── Android.mk     # NDK build configuration
│   │   │   ├── Application.mk
│   │   │   ├── main.cpp       # Main inference loop
│   │   │   ├── QnnRunner.cpp  # QNN SDK wrapper
│   │   │   └── QnnRunner.h
│   │   └── assets/            # Model binaries
│   └── output/                # Compiled executables
├── data/                      # Vector datasets
│   ├── siftsmall/            # 10k vectors
│   └── sift/                 # 1M vectors
├── models/                    # ONNX models
├── prepare/
│   └── create_model.py       # ONNX model generation
├── qnn/                       # QNN conversion
│   ├── convert_to_qnn.sh     # Conversion script
│   ├── htp_config.json       # HTP configuration
│   └── quant_overrides.json  # Quantization settings
├── results/                   # Experiment results
│   ├── siftsmall_10k/
│   └── sift_1M/
├── scripts/                   # Automation scripts
│   ├── build.sh              # Build everything
│   ├── deploy.sh             # Deploy and run
│   ├── clean.sh              # Clean artifacts
│   └── run_all_datasets.sh   # Run all experiments
├── requirements.txt           # Python dependencies
├── README.md                 # This file
```

---

## Performance

### Results Structure

Results are saved to dataset-specific directories:

```
results/
├── siftsmall_10k/
│   ├── results.txt    # Top-K results for each query
│   └── metrics.txt    # Performance metrics
└── sift_1M/
    ├── results.txt
    └── metrics.txt
```

### Metrics Captured

Each `metrics.txt` file contains:

- **Dataset Information**: Number of queries, documents, dimensions, TOP-K
- **Overall Performance**: Total execution time, throughput (queries/sec)
- **NPU Performance**: Average latency, std deviation, min/max, percentiles (P50/P95/P99)
- **CPU Post-processing**: Same metrics for distance computation and sorting
- **End-to-End Per Query**: Combined latency metrics
- **Time Breakdown**: Percentage of time spent in NPU vs CPU

### Example Metrics

```
=== QNN RAG Demo Performance Metrics ===

Dataset Information:
  Number of queries: 100
  Number of documents: 10000
  Dimension: 128
  Top-K: 5

Overall Performance:
  Total execution time: 2.456789 s
  Throughput: 40.7 queries/sec

NPU Performance:
  Average latency: 15.234 ms/query
  P50 latency: 14.987 ms
  P95 latency: 18.234 ms
  P99 latency: 19.876 ms
...
```

### View Results

```bash
# View siftsmall metrics
cat results/siftsmall_10k/metrics.txt

# View sift metrics
cat results/sift_1M/metrics.txt

# Compare throughput
grep "Throughput" results/*/metrics.txt

# Compare average latency
grep "Average latency" results/*/metrics.txt

# View top-5 results for first 10 queries
head -20 results/siftsmall_10k/results.txt
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "QNN_SDK_ROOT is not set"

**Solution**:
```bash
export QNN_SDK_ROOT=~/qualcomm/qnn
```

Add to `~/.bashrc` to make it permanent.

#### 2. "ANDROID_NDK_ROOT not found"

**Solution**:
```bash
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653
```

Verify the path exists: `ls -la $ANDROID_NDK_ROOT`

#### 3. "No device detected"

**Solution**:
```bash
# Check device connection
adb devices

# If no devices listed:
# 1. Reconnect USB cable
# 2. Enable USB debugging on device
# 3. Accept USB debugging prompt on device
```

#### 4. "Context binary generation failed"

**Solution**:
- Try using CPU backend: Edit `scripts/deploy.sh` and change `libQnnHtp.so` to `libQnnCpu.so`
- Check device logs: `adb logcat | grep -i qnn`

#### 5. "Dimension mismatch" errors

**Solution**:
```bash
# Verify data files
python3 -c "import numpy as np; f=open('data/siftsmall/siftsmall_base.fvecs','rb'); print('Dimension:', np.frombuffer(f.read(4), dtype='int32')[0])"

# Should output: Dimension: 128
```

#### 6. Python dependency issues

**Solution**:
```bash
# Clean and reinstall
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Getting Help

If you encounter issues:

1. Check device logs:
   ```bash
   adb logcat | grep -i qnn
   ```

2. List files on device:
   ```bash
   adb shell ls -la /data/local/tmp/qnn-rag-demo
   ```

3. Check disk space:
   ```bash
   adb shell df -h
   ```

4. Clean and rebuild:
   ```bash
   bash scripts/clean.sh
   bash scripts/build.sh siftsmall
   ```

---

## Cleanup

### Clean Build Artifacts

```bash
bash scripts/clean.sh
```

### Clean QNN Artifacts

```bash
rm -rf qnn/qnn_artifacts qnn/raw_inputs qnn/input_list.txt
```

### Clean Results

```bash
rm -rf results/
```

### Clean Models

```bash
rm -rf models/*.onnx
```

### Clean Device

```bash
adb shell rm -rf /data/local/tmp/qnn-rag-demo
```
