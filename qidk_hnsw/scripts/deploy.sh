#!/bin/bash
set -e

DATASET_NAME=${1:-"siftsmall"}
M=${2:-16}
TOP_K=${3:-10}
EF_SEARCH=${4:-50}

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

HOST_EXE=./android/output/qidk_rag_demo_hnsw
HNSW_INDEX=./data/${DATASET_NAME}/${DATASET_NAME}_hnsw_M${M}.bin
VECTORS_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_base.fvecs
QUERY_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_query.fvecs
MODEL_ARTIFACT_DIR=./qnn/qnn_artifacts/${DATASET_NAME}
MODEL_LIB_SO=${MODEL_ARTIFACT_DIR}/model_lib/aarch64-android/libvector_search_${DATASET_NAME}.so
MODEL_LIB_BASENAME=$(basename "$MODEL_LIB_SO")
CONTEXT_GENERATOR=$QNN_SDK_ROOT/bin/aarch64-android/qnn-context-binary-generator

QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
BUILT_LIBS_DIR=./android/app/main/libs/arm64-v8a

QNN_LIBS=("libQnnHtp.so" "libQnnSystem.so" "libc++_shared.so" "libQnnHtpPrepare.so")

REMOTE_DIR=/data/local/tmp/qnn-hnsw-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo_hnsw
REMOTE_INDEX=$REMOTE_DIR/hnsw_index.bin
REMOTE_VECTORS=$REMOTE_DIR/vectors.fvecs
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_CONTEXT=$REMOTE_DIR/context.bin
REMOTE_RESULTS_DIR=$REMOTE_DIR/results
REMOTE_MODEL_LIB=$REMOTE_DIR/$MODEL_LIB_BASENAME

LOCAL_RESULTS_DIR=./results/${DATASET_NAME}_M${M}_ef${EF_SEARCH}

echo "=== Deploying HNSW on Snapdragon NPU ==="
echo "  Dataset: $DATASET_NAME"
echo "  HNSW M: $M, ef_search: $EF_SEARCH, top_k: $TOP_K"
echo "  Results: $LOCAL_RESULTS_DIR"
echo ""

if ! adb devices | grep -q "device$"; then
    echo "Error: No device detected. Connect via ADB."
    exit 1
fi

for f in "$HOST_EXE" "$HNSW_INDEX" "$VECTORS_FILE" "$QUERY_FILE" "$MODEL_LIB_SO" "$CONTEXT_GENERATOR"; do
    if [ ! -f "$f" ]; then
        echo "Error: Missing file: $f"
        exit 1
    fi
done

echo "[1/6] Creating remote directory..."
adb shell "mkdir -p $REMOTE_DIR" 2>/dev/null

echo "[2/6] Pushing executable and data..."
adb push $HOST_EXE $REMOTE_EXE >/dev/null 2>&1
adb shell "chmod +x $REMOTE_EXE"
adb push $HNSW_INDEX $REMOTE_INDEX >/dev/null 2>&1
adb push $VECTORS_FILE $REMOTE_VECTORS >/dev/null 2>&1
adb push $QUERY_FILE $REMOTE_QUERY >/dev/null 2>&1

echo "[3/6] Pushing QNN model library..."
adb push "$MODEL_LIB_SO" "$REMOTE_MODEL_LIB" >/dev/null 2>&1
adb push "$CONTEXT_GENERATOR" $REMOTE_DIR/ >/dev/null 2>&1
adb shell "chmod +x $REMOTE_DIR/qnn-context-binary-generator"

echo "[4/6] Pushing QNN runtime libraries..."
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$QNN_LIBS_DIR/$lib" ]; then
        adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null 2>&1
    elif [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        adb push "$BUILT_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null 2>&1
    fi
done

SKEL_PUSHED=false
HEXAGON_DIRS=(
    "$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned"
)
ADSP_LIBS=("libQnnHtpV73Skel.so" "libQnnHtpV75Skel.so" "libQnnHtpV79Skel.so")

for dir in "${HEXAGON_DIRS[@]}"; do
    for lib in "${ADSP_LIBS[@]}"; do
        if [ -f "$dir/$lib" ]; then
            adb push "$dir/$lib" $REMOTE_DIR/ >/dev/null 2>&1
            SKEL_PUSHED=true
        fi
    done
done

echo "[5/6] Generating context binary on device..."
adb shell "cd $REMOTE_DIR && \
           rm -rf output context.bin && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qnn-context-binary-generator --model $MODEL_LIB_BASENAME --backend libQnnHtp.so --binary_file context" >/dev/null 2>&1

if adb shell "[ -f $REMOTE_DIR/output/context.bin ]" >/dev/null 2>&1; then
    adb shell "cd $REMOTE_DIR && cp output/context.bin context.bin"
else
    echo "Error: Failed to generate context binary on device"
    exit 1
fi

echo "[6/6] Running HNSW search on NPU..."
echo ""

adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           export ADSP_LIBRARY_PATH_64=. && \
           ./qidk_rag_demo_hnsw hnsw_index.bin vectors.fvecs query.fvecs results context.bin libQnnHtp.so $TOP_K $EF_SEARCH"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=== Search Completed Successfully ==="
    
    mkdir -p "$LOCAL_RESULTS_DIR"
    
    adb pull $REMOTE_RESULTS_DIR/results.txt "$LOCAL_RESULTS_DIR/results.txt" >/dev/null 2>&1
    adb pull $REMOTE_RESULTS_DIR/metrics.txt "$LOCAL_RESULTS_DIR/metrics.txt" >/dev/null 2>&1
    
    if [ -f "$LOCAL_RESULTS_DIR/metrics.txt" ]; then
        echo ""
        cat "$LOCAL_RESULTS_DIR/metrics.txt"
        echo ""
        echo "Full results saved to: $LOCAL_RESULTS_DIR/"
    fi
else
    echo ""
    echo "Error: Search failed with exit code $RESULT"
    exit 1
fi

echo ""
echo "Cleanup device: adb shell rm -rf $REMOTE_DIR"
