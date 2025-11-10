#!/bin/bash
set -e

# --- CONFIGURATION ---
: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"
# --- END CONFIGURATION ---

# Detect ndk-build
if [ -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/ndk-build"
elif [ -f "$ANDROID_NDK_ROOT/android-ndk-r25c/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/android-ndk-r25c/ndk-build"
    ANDROID_NDK_ROOT="$ANDROID_NDK_ROOT/android-ndk-r25c"
else
    echo "Error: ndk-build not found in $ANDROID_NDK_ROOT"
    exit 1
fi

export ANDROID_NDK_ROOT
export QNN_SDK_ROOT

# Validate SDK paths
if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT not found at $ANDROID_NDK_ROOT"
    exit 1
fi
if [ ! -d "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT not found at $QNN_SDK_ROOT"
    exit 1
fi

echo "--- Using ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "--- Using QNN_SDK_ROOT: $QNN_SDK_ROOT"

# Python environment setup
VENV_DIR=./.venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating Python environment and installing dependencies..."
source $VENV_DIR/bin/activate
pip install -q "numpy<1.24" "onnx==1.13.1" "protobuf==3.20.3" "pyyaml==6.0" "packaging==21.3" requests pandas

# Data and model preparation
echo "--- Running Data/Model Prep Scripts ---"
echo "Skipping data download. Using local files in data/."
python3 prepare/create_model.py

# Convert model to QNN format
echo "--- Running QNN Conversion ---"
bash qnn/convert_to_qnn.sh

# Build C++ executable
echo "--- Building C++ Host Executable ---"
pushd android/app/main/jni
echo "Cleaning previous build..."
$NDK_BUILD clean
echo "Starting new build..."
$NDK_BUILD QNN_SDK_ROOT=$QNN_SDK_ROOT
popd

# Copy output binary
mkdir -p android/output/
cp android/app/main/libs/arm64-v8a/qidk_rag_demo android/output/

echo "--- Build Complete ---"
echo "Executable: android/output/qidk_rag_demo"
echo "Model binary: android/app/src/main/assets/vector_search_10k.bin"