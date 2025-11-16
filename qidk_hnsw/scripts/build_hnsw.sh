#!/bin/bash
set -e

# Build script for HNSW-accelerated version
# Usage: ./build_hnsw.sh [dataset_name] [M] [ef_construction]

: "${ANDROID_NDK_ROOT:=$HOME/Android/Sdk/ndk/25.2.9519653}"
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

DATASET_NAME=${1:-"siftsmall"}
M=${2:-16}
EF_CONSTRUCTION=${3:-200}

echo "=== Building HNSW QNN RAG Demo ==="
echo "  Dataset: $DATASET_NAME"
echo "  HNSW M: $M"
echo "  ef_construction: $EF_CONSTRUCTION"
echo ""

# Detect ndk-build
if [ -f "$ANDROID_NDK_ROOT/ndk-build" ]; then
    NDK_BUILD="$ANDROID_NDK_ROOT/ndk-build"
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


# Build HNSW index
echo "--- Building HNSW Index ---"
BASE_FILE="data/${DATASET_NAME}/${DATASET_NAME}_base.fvecs"
INDEX_FILE="data/${DATASET_NAME}/${DATASET_NAME}_hnsw_M${M}.bin"

if [ ! -f "$BASE_FILE" ]; then
    echo "ERROR: Base vectors file not found: $BASE_FILE"
    exit 1
fi

if [ -f "$INDEX_FILE" ]; then
    echo "HNSW index already exists: $INDEX_FILE"
    read -p "Rebuild? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping HNSW index build."
    else
        python3 build_hnsw_index.py $DATASET_NAME $M $EF_CONSTRUCTION
    fi
else
    python3 build_hnsw_index.py $DATASET_NAME $M $EF_CONSTRUCTION
fi

# Update Android.mk to include HNSW sources
echo "--- Updating Android.mk for HNSW ---"
ANDROID_MK="android/app/main/jni/Android.mk"

# Backup original
cp $ANDROID_MK ${ANDROID_MK}.backup

# Add hnsw_search.cpp to LOCAL_SRC_FILES
sed -i 's/LOCAL_SRC_FILES := \\/LOCAL_SRC_FILES := \\/' $ANDROID_MK
sed -i '/main.cpp/a\    hnsw_search.cpp \\' $ANDROID_MK

echo "[OK] Android.mk updated for HNSW"

# Build C++ executable
echo "--- Building C++ HNSW Executable ---"
pushd android/app/main/jni

# Copy HNSW sources to jni directory
cp ../../../../hnsw_search.h .
cp ../../../../hnsw_search.cpp .
cp ../../../../hnsw_main.cpp main.cpp

echo "Cleaning previous build..."
$NDK_BUILD clean
echo "Starting new build..."
$NDK_BUILD QNN_SDK_ROOT=$QNN_SDK_ROOT

popd

# Restore Android.mk
mv ${ANDROID_MK}.backup $ANDROID_MK

# Copy output binary
mkdir -p android/output/
cp android/app/main/libs/arm64-v8a/qidk_rag_demo android/output/qidk_rag_demo_hnsw

echo "--- Build Complete ---"
echo "Executable: android/output/qidk_rag_demo_hnsw"
echo "HNSW Index: $INDEX_FILE"
echo ""
echo "Deploy with:"
echo "  bash scripts/deploy_hnsw.sh $DATASET_NAME $M 10 50"
echo "  (top_k=10, ef_search=50)"