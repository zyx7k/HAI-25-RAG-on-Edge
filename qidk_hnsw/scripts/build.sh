#!/bin/bash
set -e

DATASET_NAME=${1:-"siftsmall"}
M=${2:-16}
EF_CONSTRUCTION=${3:-200}

: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653/android-ndk-r25c}"
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

echo "=== Building HNSW QNN Project ==="
echo "  Dataset: $DATASET_NAME"
echo "  HNSW M: $M"
echo "  ef_construction: $EF_CONSTRUCTION"
echo ""

if command -v ndk-build >/dev/null 2>&1; then
    NDK_BUILD=$(command -v ndk-build)
elif [ -x "$ANDROID_NDK_ROOT/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/ndk-build"
else
    FOUND_NDK=""
    if [ -d "$ANDROID_NDK_ROOT" ]; then
        FOUND_NDK=$(find "$ANDROID_NDK_ROOT" -maxdepth 5 -type f -name ndk-build -perm /111 -print -quit 2>/dev/null || true)
    fi
    if [ -z "$FOUND_NDK" ] && [ -d "$HOME/Android/Sdk/ndk" ]; then
        FOUND_NDK=$(find "$HOME/Android/Sdk/ndk" -maxdepth 3 -type f -name ndk-build -perm /111 -print -quit 2>/dev/null || true)
    fi
    if [ -z "$FOUND_NDK" ] && [ -n "$ANDROID_SDK_ROOT" ] && [ -d "$ANDROID_SDK_ROOT" ]; then
        FOUND_NDK=$(find "$ANDROID_SDK_ROOT" -maxdepth 4 -type f -name ndk-build -perm /111 -print -quit 2>/dev/null || true)
    fi

    if [ -n "$FOUND_NDK" ]; then
        NDK_BUILD="$FOUND_NDK"
    else
        echo "Error: ndk-build not found. Please set ANDROID_NDK_ROOT or add ndk-build to PATH."
        exit 1
    fi
fi

export ANDROID_NDK_ROOT
export QNN_SDK_ROOT

if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "Error: ANDROID_NDK_ROOT not found at $ANDROID_NDK_ROOT"
    exit 1
fi
if [ ! -d "$QNN_SDK_ROOT" ]; then
    echo "Error: QNN_SDK_ROOT not found at $QNN_SDK_ROOT"
    exit 1
fi

echo "Using ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "Using QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo ""

VENV_DIR=./.venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Installing Python dependencies..."
source $VENV_DIR/bin/activate
pip install -q -r requirements.txt 2>&1 | grep -v "WARNING" || true

echo "--- Step 1/4: Generating ONNX model ---"
python3 src/create_model.py $DATASET_NAME

echo "--- Step 2/4: Converting ONNX to QNN context binary ---"
bash qnn/convert_to_qnn.sh $DATASET_NAME

if [ ! -f "android/app/main/jni/context.bin" ]; then
    echo "Error: context.bin not found after conversion"
    exit 1
fi

echo "--- Step 3/4: Building HNSW index ---"
BASE_FILE="data/${DATASET_NAME}/${DATASET_NAME}_base.fvecs"
INDEX_FILE="data/${DATASET_NAME}/${DATASET_NAME}_hnsw_M${M}.bin"

if [ ! -f "$BASE_FILE" ]; then
    echo "Error: Base vectors file not found: $BASE_FILE"
    exit 1
fi

if [ -f "$INDEX_FILE" ]; then
    echo "HNSW index exists: $INDEX_FILE (skipping rebuild)"
else
    python3 src/build_hnsw_index.py $DATASET_NAME $M $EF_CONSTRUCTION
fi

echo "--- Step 4/4: Building native executable ---"
cd android/app/main
$NDK_BUILD clean >/dev/null 2>&1
$NDK_BUILD QNN_SDK_ROOT=$QNN_SDK_ROOT
cd ../../..

mkdir -p android/output/
cp android/app/main/libs/arm64-v8a/qidk_rag_demo android/output/qidk_rag_demo_hnsw

echo ""
echo "=== Build Complete ==="
echo "Executable: android/output/qidk_rag_demo_hnsw"
echo "HNSW Index: $INDEX_FILE"
echo "Context Binary: android/app/main/jni/context.bin"
echo ""
echo "Next: Deploy to device with:"
echo "  bash scripts/deploy.sh $DATASET_NAME $M <top_k> <ef_search>"
echo "  Example: bash scripts/deploy.sh $DATASET_NAME $M 10 50"
