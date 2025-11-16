#!/bin/bash
set -e

# Deploy and run HNSW search on device
# Usage: ./deploy_hnsw.sh [dataset_name] [M] [top_k] [ef_search]

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

DATASET_NAME=${1:-"siftsmall"}
M=${2:-16}
TOP_K=${3:-10}
EF_SEARCH=${4:-50}

HOST_EXE=./android/output/qidk_rag_demo_hnsw
HNSW_INDEX=./data/${DATASET_NAME}/${DATASET_NAME}_hnsw_M${M}.bin
VECTORS_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_base.fvecs
QUERY_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_query.fvecs

QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
BUILT_LIBS_DIR=./android/app/main/libs/arm64-v8a

QNN_LIBS=("libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnSystem.so" "libc++_shared.so")

REMOTE_DIR=/data/local/tmp/qnn-hnsw-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo_hnsw
REMOTE_INDEX=$REMOTE_DIR/hnsw_index.bin
REMOTE_VECTORS=$REMOTE_DIR/vectors.fvecs
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_RESULTS_DIR=$REMOTE_DIR/results

LOCAL_RESULTS_DIR=./results/${DATASET_NAME}_hnsw_M${M}_ef${EF_SEARCH}

echo "=== Deploying HNSW QNN Demo ==="
echo "  Dataset: $DATASET_NAME"
echo "  HNSW M: $M"
echo "  TOP_K: $TOP_K"
echo "  ef_search: $EF_SEARCH"
echo "  Results dir: $LOCAL_RESULTS_DIR"
echo ""

# Check device connection
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device detected. Connect via ADB."
    exit 1
fi
echo "[OK] Device connected"

# Validate files
for f in "$HOST_EXE" "$HNSW_INDEX" "$VECTORS_FILE" "$QUERY_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing file: $f"
        exit 1
    fi
done

# Create remote directory
adb shell "mkdir -p $REMOTE_DIR"
echo "[OK] Created $REMOTE_DIR"

# Push executable
adb push $HOST_EXE $REMOTE_EXE >/dev/null
adb shell "chmod +x $REMOTE_EXE"
echo "[OK] Executable pushed"

# Push HNSW index and data files
echo "Pushing HNSW index (this may take a moment)..."
adb push $HNSW_INDEX $REMOTE_INDEX >/dev/null
echo "[OK] HNSW index pushed"

echo "Pushing vectors..."
adb push $VECTORS_FILE $REMOTE_VECTORS >/dev/null
echo "[OK] Vectors pushed"

adb push $QUERY_FILE $REMOTE_QUERY >/dev/null
echo "[OK] Query file pushed"

# Push QNN libraries
echo "Pushing QNN libraries..."
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        adb push "$BUILT_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    elif [ -f "$QNN_LIBS_DIR/$lib" ]; then
        adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    else
        echo "WARNING: Missing $lib"
    fi
done
echo "[OK] Libraries pushed"

# Push optional DSP skeleton libs
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
[ "$SKEL_PUSHED" = true ] && echo "[OK] NPU skeleton libraries pushed"

# Run HNSW search on device
echo ""
echo "--- Running HNSW Search on Device ---"
echo "Parameters: TOP_K=$TOP_K, ef_search=$EF_SEARCH"
echo ""

adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qidk_rag_demo_hnsw hnsw_index.bin vectors.fvecs query.fvecs results libQnnHtp.so $TOP_K $EF_SEARCH"
RESULT=$?

# Retrieve results
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "[OK] Search completed successfully!"
    
    mkdir -p "$LOCAL_RESULTS_DIR"
    
    adb pull $REMOTE_RESULTS_DIR/results.txt "$LOCAL_RESULTS_DIR/results.txt" >/dev/null 2>&1
    adb pull $REMOTE_RESULTS_DIR/metrics.txt "$LOCAL_RESULTS_DIR/metrics.txt" >/dev/null 2>&1
    
    if [ -f "$LOCAL_RESULTS_DIR/metrics.txt" ]; then
        echo "[OK] Results saved to $LOCAL_RESULTS_DIR/"
        echo ""
        echo "=== Performance Summary ==="
        grep -E "Number of queries:|Throughput:|Average latency:|P95 latency:" "$LOCAL_RESULTS_DIR/metrics.txt"
        echo ""
        echo "Full metrics: $LOCAL_RESULTS_DIR/metrics.txt"
        echo "Full results: $LOCAL_RESULTS_DIR/results.txt"
    else
        echo "WARNING: Could not retrieve metrics"
    fi
    
    echo ""
    echo "Compare with brute-force:"
    if [ -f "results/${DATASET_NAME}_*/metrics.txt" ]; then
        echo "  Brute-force metrics: results/${DATASET_NAME}_*/metrics.txt"
    fi
else
    echo ""
    echo "[ERROR] Search failed with code $RESULT"
    echo ""
    echo "Debug steps:"
    echo "  adb logcat | grep -i hnsw"
    echo "  adb shell ls -la $REMOTE_DIR"
    exit 1
fi

echo ""
echo "Cleanup:"
echo "  adb shell rm -rf $REMOTE_DIR"