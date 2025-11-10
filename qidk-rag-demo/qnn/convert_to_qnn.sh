#!/bin/bash
set -e

# --- Configuration ---
ONNX_MODEL=models/vector_search_10k.onnx
QUERY_DATA_FILE=data/siftsmall_query.fvecs
FINAL_OUTPUT_BIN=android/app/src/main/assets/vector_search_10k.bin
# --- End Configuration ---

# Ensure QNN SDK path is set
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT is not set."
    echo "Please set it (e.g., export QNN_SDK_ROOT=~/qualcomm/qnn)"
    exit 1
fi
# Ensure NDK path is set (needed by the lib generator)
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: ANDROID_NDK_ROOT is not set."
    echo "Please set it (e.g., export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/25.2.9519653)"
    exit 1
fi

# Source QNN environment
source "$QNN_SDK_ROOT/bin/envsetup.sh"

# Add NDK to the PATH for qnn-model-lib-generator
export PATH=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

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

echo "--- Running QNN Model Lib Generator (Step 2) ---"
# This step builds the model lib for the HOST (x86_64)
qnn-model-lib-generator \
    -c "$ARTIFACTS_DIR/model.cpp" \
    -b "$ARTIFACTS_DIR/model.bin" \
    -o "$ARTIFACTS_DIR" \
    -t x86_64-linux-clang

echo "--- Running QNN Context Binary Generator (Step 3) ---"

# Add the *new host* directory to the library path
MODEL_LIB_DIR="$(pwd)/$ARTIFACTS_DIR/x86_64-linux-clang"
export LD_LIBRARY_PATH=$MODEL_LIB_DIR:$LD_LIBRARY_PATH

# *** THIS IS THE FIX ***
# 1. Point --config_file to the standard examples path.
# 2. Add --platform_options to specify your exact chip.
qnn-context-binary-generator \
    --model "$MODEL_LIB_DIR/libmodel.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --binary_file "$ARTIFACTS_DIR/npu_model.bin" \
    --config_file "$QNN_SDK_ROOT/examples/QNN/common/backend_extensions/htp_config.json" \
    --platform_options "htp.socModel:sm8550"

# Unset the path just to be clean
export LD_LIBRARY_PATH=""

# Copy the *newly generated* NPU model binary
mkdir -p "$(dirname "$FINAL_OUTPUT_BIN")"
cp "$ARTIFACTS_DIR/npu_model.bin" "$FINAL_OUTPUT_BIN"

echo "✓ NPU model (npu_model.bin) copied to $FINAL_OUTPUT_BIN"
echo "✓ Model ready for execution on Hexagon NPU"
echo "Cleaning up temporary artifacts..."
rm -rf "$RAW_DIR" "$QUANT_INPUT_LIST" "$ARTIFACTS_DIR"

echo "--- Conversion complete ---"