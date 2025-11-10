#!/bin/bash
set -e
# --- CONFIGURE THESE PATHS ---
# Assumes Android NDK is in ~/Android/Sdk/ndk/
: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"
# Assumes QAIED SDK is in ~/qualcomm/qai-direct-sdk
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"
# --- END CONFIGURATION ---

# Auto-detect ndk-build location
if [ -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/ndk-build"
elif [ -f "$ANDROID_NDK_ROOT/android-ndk-r25c/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/android-ndk-r25c/ndk-build"
    ANDROID_NDK_ROOT="$ANDROID_NDK_ROOT/android-ndk-r25c"
else
    echo "Error: ndk-build not found in $ANDROID_NDK_ROOT"
    echo "Please check your NDK installation"
    exit 1
fi

export ANDROID_NDK_ROOT
export QNN_SDK_ROOT
# 1. Check if SDKs exist
if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT not found at $ANDROID_NDK_ROOT"
    echo "Please edit scripts/build.sh"
    exit 1
fi
if [ ! -d "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT not found at $QNN_SDK_ROOT"
    echo "Please edit scripts/build.sh"
    exit 1
fi
echo "--- Using ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "--- Using QNN_SDK_ROOT: $QNN_SDK_ROOT"
# 2. Set up Python environment
VENV_DIR=./.venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi
echo "Activating Python environment and installing dependencies..."
source $VENV_DIR/bin/activate
# --- MODIFICATION ---
# No requirements.txt found, so manually pinning to known compatible versions.
# This is the key fix for the 'AttributeError'
pip install -q "numpy<1.24" "onnx==1.13.1" "protobuf==3.20.3" "pyyaml==6.0" "packaging==21.3" requests pandas
# --- END MODIFICATION ---
# 3. Prepare data and models
echo "--- Running Data/Model Prep Scripts ---"
# We assume data is already in `data/`, so we skip the download script.
# python3 prepare/download_data.py
echo "Skipping data download. Using local files in data/."
python3 prepare/create_model.py
# 4. Convert model to QNN format
echo "--- Running QNN Conversion ---"
# This script sources the QNN env and runs the converter tools
bash qnn/convert_to_qnn.sh

# 5. Build the C++ executable
echo "--- Building C++ Host Executable ---"
# We need to pass QNN_SDK_ROOT to the makefile
# ndk-build is picky about where it's run
# FIX: Correct path to JNI directory
pushd android/app/main/jni

# --- ✅ ADDED CLEAN STEP ---
echo "Cleaning previous build..."
$NDK_BUILD clean
# --- END OF ADDED STEP ---

echo "Starting new build..."
$NDK_BUILD QNN_SDK_ROOT=$QNN_SDK_ROOT
popd

# 6. Copy executable to a clean location
mkdir -p android/output/
cp android/app/main/libs/arm64-v8a/qidk_rag_demo android/output/
echo "--- Build Complete ---"
echo "Executable is at: android/output/qidk_rag_demo"
echo "Model binary is at: android/app/src/main/assets/vector_search_10k.bin"