#!/bin/bash
set -e

# Usage: ./build.sh [dataset_name] [model_size_suffix]
# Example: ./build.sh siftsmall 10k
# Example: ./build.sh sift 1M

# --- CONFIGURATION ---
: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

# Parse arguments
DATASET_NAME=${1:-"siftsmall"}
MODEL_SIZE_SUFFIX=${2:-""}

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

echo "=== Building QNN RAG Demo ==="
echo "  Dataset: $DATASET_NAME"
echo "  Model size: $MODEL_SIZE_SUFFIX"
echo ""
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
pip install -q -r requirements.txt 2>&1 | grep -v "WARNING" || true

# Data and model preparation
echo "--- Running Data/Model Prep Scripts ---"
echo "Skipping data download. Using local files in data/."
python3 prepare/create_model.py $DATASET_NAME

# Convert model to QNN format
echo "--- Running QNN Conversion ---"
bash qnn/convert_to_qnn.sh $DATASET_NAME $MODEL_SIZE_SUFFIX

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
echo "Model binary: android/app/src/main/assets/vector_search_${MODEL_SIZE_SUFFIX}.bin"