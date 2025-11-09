#!/bin/bash
set -e
# --- Configuration ---
# Set to true/false to match create_model.py
USE_FULL_DATASET=false
if [ "$USE_FULL_DATASET" = true ]; then
    ONNX_MODEL=models/vector_search_1M.onnx
    QUERY_DATA_FILE=data/siftsmall_query.fvecs
    OUTPUT_BIN=android/app/src/main/assets/vector_search_1M.bin
else
    ONNX_MODEL=models/vector_search_10k.onnx
    QUERY_DATA_FILE=data/siftsmall_query.fvecs
    OUTPUT_BIN=android/app/src/main/assets/vector_search_10k.bin
fi
# This script must be run from the root qidk-rag-demo/ directory
# It requires QNN_SDK_ROOT to be set
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT is not set."
    echo "Please set it (e.g., export QNN_SDK_ROOT=~/qualcomm/qai-direct-sdk)"
    exit 1
fi
# This file will list the raw query vectors for quantization
QUANT_INPUT_LIST=./qnn/input_list.txt
# 1. Source the QNN SDK environment
source $QNN_SDK_ROOT/bin/envsetup.sh

# FIX: Ensure qnn_artifacts is a directory, not a file
if [ -f "./qnn/qnn_artifacts" ]; then
    echo "Removing qnn_artifacts file..."
    rm ./qnn/qnn_artifacts
fi
mkdir -p ./qnn/qnn_artifacts

echo "Creating representative dataset for quantization..."
# 2. Create the quantization input list
#    We need to convert the .fvecs query file to raw float32 binaries
#    and list them in input_list.txt.
#    The QNN tools expect one raw file per input.
RAW_DIR=./qnn/raw_inputs
rm -rf $RAW_DIR $QUANT_INPUT_LIST
mkdir -p $RAW_DIR
# Use a python script to do this conversion
python3 -c "
import numpy as np; import sys;
sys.path.append('./prepare'); 
from download_data import read_fvecs;
queries = read_fvecs('$QUERY_DATA_FILE');
input_list_file = open('$QUANT_INPUT_LIST', 'w');
for i, q in enumerate(queries):
    raw_path = f'$RAW_DIR/query_{i}.raw';
    q.astype('float32').tofile(raw_path);
    input_list_file.write('query:=' + raw_path + '\n');
input_list_file.close();
print(f'Wrote {len(queries)} raw query files to $RAW_DIR');
"
echo "Representative dataset created."
echo "Running qnn-onnx-converter..."
# 3. Run the ONNX Converter
#    This converts the .onnx file and quantizes it using the
#    representative dataset. It outputs a QNN model .cpp and .bin
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --input_list "$QUANT_INPUT_LIST" \
    --output_path ./qnn/qnn_artifacts/model \
    --quantization_overrides ./qnn/quant_overrides.json
echo "ONNX model converted."

echo "Using quantized model binary for HTP (Hexagon NPU)..."
# 4. Copy the quantized model binary to assets
#    The model.bin from qnn-onnx-converter is already quantized and ready for HTP
mkdir -p $(dirname $OUTPUT_BIN)
cp ./qnn/qnn_artifacts/model.bin $OUTPUT_BIN

echo "Quantized model copied to $OUTPUT_BIN"
echo "Note: This model will run on Hexagon NPU when loaded with QNN HTP backend on device."

echo "QNN Context Binary created at $OUTPUT_BIN"
echo "Cleaning up artifacts..."
rm -rf ./qnn/raw_inputs ./qnn/input_list.txt ./qnn/qnn_artifacts
echo "Conversion complete - model is compiled for Hexagon NPU acceleration!"