#!/bin/bash
set -e

: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

# Parse arguments
DATASET_NAME=${1:-"siftsmall"}
MODEL_SIZE_SUFFIX=${2:-""}
BATCH_SIZE=${3:-1}

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

# Determine model filename based on batch size
if [ "$BATCH_SIZE" -gt 1 ]; then
    MODEL_SUFFIX="${MODEL_SIZE_SUFFIX}_b${BATCH_SIZE}"
else
    MODEL_SUFFIX="${MODEL_SIZE_SUFFIX}"
fi

echo "Building: $DATASET_NAME, batch=$BATCH_SIZE"

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

VENV_DIR=./.venv
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
pip install -q -r requirements.txt 2>&1 | grep -v "WARNING" || true

python3 prepare/create_model.py $DATASET_NAME -1 $BATCH_SIZE

bash qnn/convert_to_qnn.sh $DATASET_NAME $MODEL_SIZE_SUFFIX $BATCH_SIZE

pushd android/app/main/jni
$NDK_BUILD clean
$NDK_BUILD QNN_SDK_ROOT=$QNN_SDK_ROOT
popd

mkdir -p android/output/
cp android/app/main/libs/arm64-v8a/qidk_rag_demo android/output/

echo "Build complete: android/output/qidk_rag_demo"