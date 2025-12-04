#!/bin/bash
set -e

if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT not set"
    exit 1
fi

: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"

if [ -f "$ANDROID_NDK_ROOT/android-ndk-r25c/ndk-build" ]; then
    ANDROID_NDK_ROOT="$ANDROID_NDK_ROOT/android-ndk-r25c"
fi

if [ ! -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
    NDK_BUILD_PATH=$(find $HOME -name "ndk-build" -type f 2>/dev/null | head -1)
    if [ -n "$NDK_BUILD_PATH" ]; then
        ANDROID_NDK_ROOT=$(dirname "$NDK_BUILD_PATH")
    fi
fi

if [ ! -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
    echo "ERROR: ANDROID_NDK_ROOT not set or ndk-build not found"
    echo "Please set ANDROID_NDK_ROOT to your NDK installation"
    exit 1
fi
export PATH="$ANDROID_NDK_ROOT:$PATH"

source "$QNN_SDK_ROOT/bin/envsetup.sh"

pip show onnx >/dev/null 2>&1 || pip install onnx==1.13.1 protobuf==3.20.3

DATASET=${1:-siftsmall}
BATCH_SIZE=${2:-1}
REORDERED=${3:-}

if [ "$REORDERED" = "reordered" ]; then
    INDEX_DIR="models/ivf_${DATASET}_reordered"
else
    INDEX_DIR="models/ivf_${DATASET}"
fi
ONNX_MODEL="${INDEX_DIR}/centroids.onnx"
OUTPUT_DIR="${INDEX_DIR}"

echo "Converting: ${ONNX_MODEL} -> ${OUTPUT_DIR}/centroids.bin"

rm -rf "${OUTPUT_DIR}/centroids_model.cpp" "${OUTPUT_DIR}/centroids_model.bin" "${OUTPUT_DIR}/x86_64-linux-clang" "${OUTPUT_DIR}/aarch64-android"

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network "${ONNX_MODEL}" \
    --output_path "${OUTPUT_DIR}/centroids_model" \
    --batch "${BATCH_SIZE}"

# Rename the output file to have .cpp extension (QNN generates without extension)
if [ -f "${OUTPUT_DIR}/centroids_model" ] && [ ! -f "${OUTPUT_DIR}/centroids_model.cpp" ]; then
    mv "${OUTPUT_DIR}/centroids_model" "${OUTPUT_DIR}/centroids_model.cpp"
fi

# Patch batch size in generated C++ if needed
if [ "$BATCH_SIZE" -gt 1 ]; then
    sed -i "s/dimensions_query\[\] = {1, 128}/dimensions_query[] = {${BATCH_SIZE}, 128}/g" "${OUTPUT_DIR}/centroids_model.cpp"
    sed -i "s/dimensions_scores\[\] = {1, 1024}/dimensions_scores[] = {${BATCH_SIZE}, 1024}/g" "${OUTPUT_DIR}/centroids_model.cpp"
fi

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c "${OUTPUT_DIR}/centroids_model.cpp" \
    -b "${OUTPUT_DIR}/centroids_model.bin" \
    -o "${OUTPUT_DIR}" \
    -t x86_64-linux-clang

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c "${OUTPUT_DIR}/centroids_model.cpp" \
    -b "${OUTPUT_DIR}/centroids_model.bin" \
    -o "${OUTPUT_DIR}" \
    -t aarch64-android

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --model "${OUTPUT_DIR}/x86_64-linux-clang/libcentroids_model.so" \
    --backend "${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "${OUTPUT_DIR}" \
    --binary_file "centroids"

if [ -f "${OUTPUT_DIR}/centroids.bin" ]; then
    echo "Created: ${OUTPUT_DIR}/centroids.bin"
elif [ -f "${OUTPUT_DIR}/centroids.bin.bin" ]; then
    mv "${OUTPUT_DIR}/centroids.bin.bin" "${OUTPUT_DIR}/centroids.bin"
    echo "Created: ${OUTPUT_DIR}/centroids.bin"
fi
