#!/bin/bash
set -euo pipefail

# Convert ONNX model to QNN context binary for HTP execution
# Usage: ./convert_to_qnn.sh <dataset_name> [model_size_suffix]

DATASET=${1:-siftsmall}
MODEL_SUFFIX=${2:-}

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"
: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk}"

ONNX_MODEL="android/app/main/jni/vector_search_${DATASET}${MODEL_SUFFIX}.onnx"
CONTEXT_OUT="android/app/main/jni/context.bin"
ARTIFACT_ROOT="qnn/qnn_artifacts/${DATASET}${MODEL_SUFFIX}"
LIB_NAME="vector_search_${DATASET}${MODEL_SUFFIX}"
MODEL_PREFIX="${ARTIFACT_ROOT}/${LIB_NAME}"
MODEL_LIB_DIR="${ARTIFACT_ROOT}/model_lib"

log() {
    echo "[convert_to_qnn] $*"
}

fail() {
    echo "[convert_to_qnn][ERROR] $*" >&2
    exit 1
}

ensure_file() {
    local path="$1"
    local msg="$2"
    [ -e "$path" ] || fail "$msg ($path)"
}

log "Dataset: $DATASET${MODEL_SUFFIX}"
log "Using QNN_SDK_ROOT=$QNN_SDK_ROOT"
log "Using ANDROID_NDK_ROOT=$ANDROID_NDK_ROOT"

ensure_file "$ONNX_MODEL" "ONNX model not found"
ensure_file "$QNN_SDK_ROOT/bin/envsetup.sh" "QNN envsetup script missing"

# Prepare directories
rm -rf "$ARTIFACT_ROOT" "$MODEL_LIB_DIR"
mkdir -p "$ARTIFACT_ROOT"

# Activate QNN environment (adds host binaries + python packages to PATH)
source "$QNN_SDK_ROOT/bin/envsetup.sh" >/dev/null

# Ensure ndk-build is on PATH for qnn-model-lib-generator
if ! command -v ndk-build >/dev/null 2>&1; then
    CANDIDATE=""
    if [ -x "$ANDROID_NDK_ROOT/ndk-build" ]; then
        CANDIDATE="$ANDROID_NDK_ROOT"
    else
        FOUND=$(find "$ANDROID_NDK_ROOT" -maxdepth 3 -name ndk-build -perm /111 -print -quit 2>/dev/null || true)
        CANDIDATE=$(dirname "$FOUND")
    fi
    if [ -n "$CANDIDATE" ]; then
        export PATH="$CANDIDATE:$PATH"
    fi
fi

if ! command -v ndk-build >/dev/null 2>&1; then
    fail "ndk-build not found. Please ensure ANDROID_NDK_ROOT points to the NDK installation."
fi

command -v qnn-onnx-converter >/dev/null 2>&1 || fail "qnn-onnx-converter not found in PATH"
command -v qnn-model-lib-generator >/dev/null 2>&1 || fail "qnn-model-lib-generator not found in PATH"
command -v qnn-context-binary-generator >/dev/null 2>&1 || fail "qnn-context-binary-generator not found in PATH"

CALIB_DIR="qnn/calibration/${DATASET}${MODEL_SUFFIX}"
INPUT_LIST="$CALIB_DIR/input_list.txt"

log "Step 1/5: Generating quantization calibration data"
python3 src/generate_input_list.py "$DATASET" || fail "Failed to generate calibration data"
ensure_file "$INPUT_LIST" "Calibration input list not generated"

log "Step 2/5: Converting ONNX -> QNN IR (quantized INT8)"
qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --output_path "$MODEL_PREFIX" \
    --input_list "$INPUT_LIST" \
    --param_quantizer symmetric \
    --act_quantizer symmetric \
    --weights_bitwidth 8 \
    --act_bitwidth 8 \
    --float_bitwidth 32 \
    --model_version 1.0

# The converter emits a file without extension. Rename to .cpp for the lib generator.
if [ -f "$MODEL_PREFIX" ] && [ ! -f "${MODEL_PREFIX}.cpp" ]; then
    mv "$MODEL_PREFIX" "${MODEL_PREFIX}.cpp"
fi

CPP_FILE="${MODEL_PREFIX}.cpp"
BIN_FILE="${MODEL_PREFIX}.bin"
ensure_file "$CPP_FILE" "Converter did not produce expected .cpp file"
ensure_file "$BIN_FILE" "Converter did not produce expected .bin file"

log "Step 3/5: Building QNN model libraries (host + device)"
rm -rf "$MODEL_LIB_DIR"
qnn-model-lib-generator \
    -c "$CPP_FILE" \
    -b "$BIN_FILE" \
    -t x86_64-linux-clang aarch64-android \
    -o "$MODEL_LIB_DIR" \
    -l "$LIB_NAME"

HOST_LIB="$MODEL_LIB_DIR/x86_64-linux-clang/lib${LIB_NAME}.so"
TARGET_LIB="$MODEL_LIB_DIR/aarch64-android/lib${LIB_NAME}.so"
ensure_file "$HOST_LIB" "Host model library missing after generation"
ensure_file "$TARGET_LIB" "Device model library missing after generation"

log "Step 4/5: Generating HTP context binary"
BACKEND_HOST_LIB="$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"
ensure_file "$BACKEND_HOST_LIB" "Host HTP backend library not found"

CONTEXT_BUILD_DIR="$ARTIFACT_ROOT/context_build"
mkdir -p "$CONTEXT_BUILD_DIR"
CONTEXT_STEM="${LIB_NAME}_htp_context"
qnn-context-binary-generator \
    --model "$HOST_LIB" \
    --backend "$BACKEND_HOST_LIB" \
    --output_dir "$CONTEXT_BUILD_DIR" \
    --binary_file "$CONTEXT_STEM"

GENERATED_CONTEXT="$CONTEXT_BUILD_DIR/${CONTEXT_STEM}.bin"
ensure_file "$GENERATED_CONTEXT" "Context generator did not create ${GENERATED_CONTEXT}"
cp "$GENERATED_CONTEXT" "$CONTEXT_OUT"

log "Step 5/5: Context ready -> $CONTEXT_OUT"
log "Artifacts saved under $ARTIFACT_ROOT"
