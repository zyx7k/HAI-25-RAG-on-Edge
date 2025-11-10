#!/bin/bash
set -e

# --- Configuration ---
ONNX_MODEL=models/vector_search_10k.onnx
QUERY_DATA_FILE=data/siftsmall_query.fvecs
OUTPUT_BIN=android/app/src/main/assets/vector_search_10k.bin
# --- End Configuration ---

# Ensure QNN SDK path is set
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT is not set."
    echo "Please set it (e.g., export QNN_SDK_ROOT=~/qualcomm/qnn)"
    exit 1
fi

# Source QNN environment
source "$QNN_SDK_ROOT/bin/envsetup.sh"

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

echo "--- Running QNN ONNX Converter ---"
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --input_list "$QUANT_INPUT_LIST" \
    --output_path "$ARTIFACTS_DIR/model" \
    --quantization_overrides ./qnn/quant_overrides.json

# Copy quantized model binary
mkdir -p "$(dirname "$OUTPUT_BIN")"
cp "$ARTIFACTS_DIR/model.bin" "$OUTPUT_BIN"

echo "✓ Quantized model copied to $OUTPUT_BIN"
echo "✓ Model ready for execution on Hexagon NPU"
echo "Cleaning up temporary artifacts..."
rm -rf "$RAW_DIR" "$QUANT_INPUT_LIST" "$ARTIFACTS_DIR"

echo "--- Conversion complete ---"
