#!/bin/bash
set -e

QNN_SDK_ROOT=${QNN_SDK_ROOT:-$HOME/qualcomm/qnn}
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-$HOME/Android/Sdk/ndk/25.2.9519653/android-ndk-r25c}

source "$QNN_SDK_ROOT/bin/envsetup.sh"
export PATH=$ANDROID_NDK_ROOT:$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

ONNX_MODEL=models/test_matmul.onnx
OUTPUT_DIR=./test_output
mkdir -p "$OUTPUT_DIR"

echo "=== Converting simple test model ==="
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --output_path "$OUTPUT_DIR/test_model.cpp" \
    --input_list test_input_list.txt

echo ""
echo "=== Generating model library for aarch64-android ==="

qnn-model-lib-generator \
    -c "$OUTPUT_DIR/test_model.cpp" \
    -b "$OUTPUT_DIR/test_model.bin" \
    -t aarch64-android \
    -l test_model \
    -o "$OUTPUT_DIR"

echo ""
echo "✓ Model library: $OUTPUT_DIR/aarch64-android/libtest_model.so"
echo "✓ Model binary: $OUTPUT_DIR/test_model.bin"
