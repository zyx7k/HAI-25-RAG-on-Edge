#!/bin/bash
set -e

# Usage: ./convert_to_qnn.sh <dataset_name> [model_size_suffix]
# Example: ./convert_to_qnn.sh siftsmall 10k
# Example: ./convert_to_qnn.sh sift 1M

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [model_size_suffix]"
    echo "  dataset_name: 'siftsmall' or 'sift'"
    echo "  model_size_suffix: optional, e.g., '10k', '1M' (defaults to auto-detect)"
    exit 1
fi

DATASET_NAME=$1
MODEL_SIZE_SUFFIX=${2:-""}

# Resolve absolute project root (so script works from anywhere)
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_ROOT"

# Auto-detect model size suffix if not provided
if [ -z "$MODEL_SIZE_SUFFIX" ]; then
    if [ "$DATASET_NAME" = "siftsmall" ]; then
        MODEL_SIZE_SUFFIX="10k"
    elif [ "$DATASET_NAME" = "sift" ]; then
        MODEL_SIZE_SUFFIX="1M"
    else
        echo "ERROR: Unknown dataset name: $DATASET_NAME"
        exit 1
    fi
fi

# --- Configuration ---
ONNX_MODEL="models/vector_search_${MODEL_SIZE_SUFFIX}.onnx"
QUERY_DATA_FILE="data/${DATASET_NAME}/${DATASET_NAME}_query.fvecs"
FINAL_OUTPUT_BIN="android/app/src/main/assets/vector_search_${MODEL_SIZE_SUFFIX}.bin"
# --- End Configuration ---

echo "=== QNN Conversion for ${DATASET_NAME} dataset ==="
echo "  Model: $ONNX_MODEL"
echo "  Query data: $QUERY_DATA_FILE"
echo "  Output: $FINAL_OUTPUT_BIN"
echo ""

# Ensure QNN SDK path is set
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT is not set."
    echo "Please set it (e.g., export QNN_SDK_ROOT=~/qualcomm/qnn)"
    exit 1
fi

# Ensure NDK path is set
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: ANDROID_NDK_ROOT is not set."
    echo "Please set it (e.g., export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653)"
    exit 1
fi

# Validate files exist
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found: $ONNX_MODEL"
    echo "Run: python3 prepare/create_model.py $DATASET_NAME"
    exit 1
fi

if [ ! -f "$QUERY_DATA_FILE" ]; then
    echo "ERROR: Query data file not found: $QUERY_DATA_FILE"
    exit 1
fi

# Source QNN environment
source "$QNN_SDK_ROOT/bin/envsetup.sh"

# Add NDK to PATH for qnn-model-lib-generator
# Need both the NDK root (for ndk-build) and the toolchain
export PATH=$ANDROID_NDK_ROOT:$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

# Prepare directories
ARTIFACTS_DIR=./qnn/qnn_artifacts
RAW_DIR=./qnn/raw_inputs
QUANT_INPUT_LIST=./qnn/input_list.txt

rm -rf "$ARTIFACTS_DIR" "$RAW_DIR" "$QUANT_INPUT_LIST"
mkdir -p "$ARTIFACTS_DIR" "$RAW_DIR"

echo "--- Preparing representative dataset ---"
python3 - <<PYCODE
import numpy as np, os

QUERY_FILE = "$QUERY_DATA_FILE"
RAW_DIR = "./qnn/raw_inputs"
LIST_FILE = "./qnn/input_list.txt"

def read_fvecs(filename, count=-1):
    with open(filename, 'rb') as f:
        vectors = []
        dim = None
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            current_dim = np.frombuffer(dim_data, dtype='int32')[0]
            if dim is None:
                dim = current_dim
            vec = np.frombuffer(f.read(dim * 4), dtype='float32')
            vectors.append(vec)
            if 0 < count <= len(vectors):
                break
    return np.array(vectors, dtype='float32')

queries = read_fvecs(QUERY_FILE)
os.makedirs(RAW_DIR, exist_ok=True)
with open(LIST_FILE, 'w') as f:
    for i, q in enumerate(queries):
        path = f"{RAW_DIR}/query_{i}.raw"
        q.astype('float32').tofile(path)
        f.write(f"query:={path}\n")

print(f"Wrote {len(queries)} raw queries (dim={queries.shape[1]}) to {RAW_DIR}")
PYCODE

# --- Step 1: Convert ONNX to intermediate QNN format ---
echo "--- Running QNN ONNX Converter (Step 1) ---"
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --input_list "$QUANT_INPUT_LIST" \
    --output_path "$ARTIFACTS_DIR/model" \
    --quantization_overrides ./qnn/quant_overrides.json 2>&1 | grep -v "WARNING" || true

echo "--- Renaming generated C++ file ---"
mv "$ARTIFACTS_DIR/model" "$ARTIFACTS_DIR/model.cpp"

# --- Step 2: Generate model library for Android ARM64 ---
echo "--- Running QNN Model Lib Generator (Step 2) ---"
qnn-model-lib-generator \
    -c "$ARTIFACTS_DIR/model.cpp" \
    -b "$ARTIFACTS_DIR/model.bin" \
    -o "$ARTIFACTS_DIR" \
    -t aarch64-android 2>&1 | grep -v "WARNING" || true

# --- Step 3: Skip context binary generation (can't run ARM64 tools on x86_64) ---
# Instead, we'll use the model.bin directly from Step 1
echo "--- Skipping Context Binary Generator (cross-platform limitation) ---"
echo "Using serialized model.bin directly from qnn-onnx-converter"

# The model.bin from qnn-onnx-converter already contains the serialized graph
# and can be used directly with QnnContext_createFromBinary on the device

# --- Step 4: Copy binary to assets ---
echo "--- Copying model to Android assets ---"
ASSETS_DIR="$PROJECT_ROOT/android/app/src/main/assets"
mkdir -p "$ASSETS_DIR"
cp "$ARTIFACTS_DIR/model.bin" "$FINAL_OUTPUT_BIN"

if [ -f "$FINAL_OUTPUT_BIN" ]; then
    echo "[OK] Model copied successfully to $FINAL_OUTPUT_BIN"
else
    echo "[ERROR] Copy failed. Check path or permissions."
    exit 1
fi

# --- Step 5: Cleanup ---
echo "Cleaning up temporary artifacts..."
rm -rf "$RAW_DIR" "$QUANT_INPUT_LIST"
# Keep qnn_artifacts for debugging

echo "--- Conversion complete ---"
echo "  Model binary: $FINAL_OUTPUT_BIN"
echo "  libmodel.so: $ARTIFACTS_DIR/aarch64-android/libmodel.so"

