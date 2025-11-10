#!/bin/bash
set -e

# --- Configuration ---
ONNX_MODEL=models/vector_search_10k.onnx
QUERY_DATA_FILE=data/siftsmall_query.fvecs
FINAL_OUTPUT_BIN=android/app/src/main/assets/vector_search_10k.bin
# --- End Configuration ---

# Resolve absolute project root (so script works from anywhere)
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_ROOT"

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
python3 - <<'PYCODE'
import numpy as np, os

DIM = 128
QUERY_FILE = "data/siftsmall_query.fvecs"
RAW_DIR = "./qnn/raw_inputs"
LIST_FILE = "./qnn/input_list.txt"

def read_fvecs(filename, count=-1):
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            dim = np.frombuffer(dim_data, dtype='int32')[0]
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

print(f"Wrote {len(queries)} raw queries to {RAW_DIR}")
PYCODE

# --- Step 1: Convert ONNX to intermediate QNN format ---
echo "--- Running QNN ONNX Converter (Step 1) ---"
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --input_list "$QUANT_INPUT_LIST" \
    --output_path "$ARTIFACTS_DIR/model" \
    --quantization_overrides ./qnn/quant_overrides.json

echo "--- Renaming generated C++ file ---"
mv "$ARTIFACTS_DIR/model" "$ARTIFACTS_DIR/model.cpp"

# --- Step 2: Generate model library for Android ARM64 ---
echo "--- Running QNN Model Lib Generator (Step 2) ---"
qnn-model-lib-generator \
    -c "$ARTIFACTS_DIR/model.cpp" \
    -b "$ARTIFACTS_DIR/model.bin" \
    -o "$ARTIFACTS_DIR" \
    -t aarch64-android

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
cp "$ARTIFACTS_DIR/model.bin" "$ASSETS_DIR/vector_search_10k.bin"

if [ -f "$ASSETS_DIR/vector_search_10k.bin" ]; then
    echo "✓ Model copied successfully to $ASSETS_DIR/vector_search_10k.bin"
else
    echo "❌ ERROR: Copy failed. Check path or permissions."
    exit 1
fi

# --- Step 5: Cleanup ---
echo "Cleaning up temporary artifacts..."
rm -rf "$RAW_DIR" "$QUANT_INPUT_LIST"
# Keep qnn_artifacts for debugging

echo "--- Conversion complete ---"
