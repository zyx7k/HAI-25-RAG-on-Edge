#!/bin/bash
set -e

# Usage: ./deploy.sh [dataset_name] [model_size_suffix] [top_k] [batch_size]
# Example: ./deploy.sh siftsmall 10k 5
# Example: ./deploy.sh sift 1M 10
# Example: ./deploy.sh sift 1M 5 32   # Batch size 32

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

# Timing helper function
get_elapsed_ms() {
    local start=$1
    local end=$2
    echo "scale=2; ($end - $start) * 1000" | bc
}

# Record start time
DEPLOY_START_TIME=$(date +%s.%N)

# Parse arguments
DATASET_NAME=${1:-"siftsmall"}
MODEL_SIZE_SUFFIX=${2:-""}
TOP_K=${3:-5}
BATCH_SIZE=${4:-1}

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

HOST_EXE=./android/output/qidk_rag_demo
MODEL_BIN=./qnn/qnn_artifacts/model.bin
DATA_BIN=./android/app/src/main/assets/vector_search_${MODEL_SUFFIX}.bin
QUERY_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_query.fvecs
DOC_FILE=./data/${DATASET_NAME}/${DATASET_NAME}_base.fvecs

QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
BUILT_LIBS_DIR=./android/app/main/libs/arm64-v8a

QNN_LIBS=("libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnSystem.so" "libc++_shared.so")
OPTIONAL_LIBS=(
    "libQnnHtpNetRunExtensions.so"
    "libQnnHtpV73Stub.so"
    "libQnnHtpV75Stub.so"
    "libQnnHtpV79Stub.so"
)

REMOTE_DIR=/data/local/tmp/qnn-rag-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo
REMOTE_MODEL_BIN=$REMOTE_DIR/model.bin
REMOTE_DATA=$REMOTE_DIR/$(basename $DATA_BIN)
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_DOC=$REMOTE_DIR/docs.fvecs
REMOTE_RESULTS_DIR=$REMOTE_DIR/results
REMOTE_CONFIG=$REMOTE_DIR/htp_config.json

LOCAL_RESULTS_DIR=./results/${DATASET_NAME}_${MODEL_SUFFIX}

echo "=== Deploying QNN RAG Demo ==="
echo "  Dataset: $DATASET_NAME"
echo "  Model size: $MODEL_SIZE_SUFFIX"
echo "  Batch size: $BATCH_SIZE"
echo "  TOP_K: $TOP_K"
echo "  Query file: $QUERY_FILE"
echo "  Doc file: $DOC_FILE"
echo "  Data binary: $DATA_BIN"
echo "  Config file: ./qnn/htp_config.json"
echo "  Results dir: $LOCAL_RESULTS_DIR"
echo ""

if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device detected. Connect via ADB."
    exit 1
fi
echo "[OK] Device connected"

for f in "$HOST_EXE" "$MODEL_BIN" "$DATA_BIN" "$QUERY_FILE" "$DOC_FILE" "./qnn/htp_config.json"; do
    [ ! -f "$f" ] && echo "ERROR: Missing $f" && exit 1
done

adb shell "mkdir -p $REMOTE_DIR"
echo "[OK] Created $REMOTE_DIR"

# Time: Push executable
PUSH_EXE_START=$(date +%s.%N)
adb push $HOST_EXE $REMOTE_EXE >/dev/null
adb shell "chmod +x $REMOTE_EXE"
PUSH_EXE_END=$(date +%s.%N)
PUSH_EXE_TIME=$(get_elapsed_ms $PUSH_EXE_START $PUSH_EXE_END)
echo "[OK] Executable pushed (${PUSH_EXE_TIME} ms)"

# Time: Push config
adb push ./qnn/htp_config.json $REMOTE_CONFIG >/dev/null
echo "[OK] Config pushed"

# Time: Push model binary
PUSH_MODEL_START=$(date +%s.%N)
adb push $MODEL_BIN $REMOTE_MODEL_BIN >/dev/null
PUSH_MODEL_END=$(date +%s.%N)
PUSH_MODEL_TIME=$(get_elapsed_ms $PUSH_MODEL_START $PUSH_MODEL_END)
echo "[OK] Model binary pushed (${PUSH_MODEL_TIME} ms)"

# Time: Push data binary (context weights)
PUSH_DATA_START=$(date +%s.%N)
adb push $DATA_BIN $REMOTE_DATA >/dev/null
PUSH_DATA_END=$(date +%s.%N)
PUSH_DATA_TIME=$(get_elapsed_ms $PUSH_DATA_START $PUSH_DATA_END)
echo "[OK] Data binary pushed (${PUSH_DATA_TIME} ms)"

# Time: Push query file
PUSH_QUERY_START=$(date +%s.%N)
adb push $QUERY_FILE $REMOTE_QUERY >/dev/null
PUSH_QUERY_END=$(date +%s.%N)
PUSH_QUERY_TIME=$(get_elapsed_ms $PUSH_QUERY_START $PUSH_QUERY_END)
echo "[OK] Query file pushed (${PUSH_QUERY_TIME} ms)"

# Time: Push document file (this is the big one!)
PUSH_DOC_START=$(date +%s.%N)
adb push $DOC_FILE $REMOTE_DOC >/dev/null
PUSH_DOC_END=$(date +%s.%N)
PUSH_DOC_TIME=$(get_elapsed_ms $PUSH_DOC_START $PUSH_DOC_END)
echo "[OK] Document embeddings pushed (${PUSH_DOC_TIME} ms)"

# Time: Push QNN libraries
PUSH_LIBS_START=$(date +%s.%N)
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$QNN_LIBS_DIR/$lib" ]; then
        adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    elif [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        adb push "$BUILT_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    else
        echo "ERROR: Cannot find $lib"
        exit 1
    fi
done

for lib in "${OPTIONAL_LIBS[@]}"; do
    [ -f "$QNN_LIBS_DIR/$lib" ] && adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
done

echo "Searching for NPU/Skel libs..."
SKEL_PUSHED=false
HEXAGON_DIRS=(
    "$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned"
)
ADSP_LIBS=("libQnnHtpV73Skel.so" "libQnnHtpV75Skel.so" "libQnnHtpV79Skel.so")

for dir in "${HEXAGON_DIRS[@]}"; do
    for skel in "${ADSP_LIBS[@]}"; do
        if [ -f "$dir/$skel" ]; then
            adb push "$dir/$skel" $REMOTE_DIR/ >/dev/null
            echo "[OK] Pushed $skel"
            SKEL_PUSHED=true
        fi
    done
done

if [ "$SKEL_PUSHED" = false ]; then
    echo "WARNING: No Hexagon skel libraries found."
fi

adb push ./qnn/qnn_artifacts/aarch64-android/libmodel.so $REMOTE_DIR/ >/dev/null

# Push qnn-context-binary-generator tool
QNN_CONTEXT_GEN="$QNN_SDK_ROOT/bin/aarch64-android/qnn-context-binary-generator"
if [ -f "$QNN_CONTEXT_GEN" ]; then
    adb push "$QNN_CONTEXT_GEN" $REMOTE_DIR/ >/dev/null
    adb shell "chmod +x $REMOTE_DIR/qnn-context-binary-generator"
    echo "[OK] Pushed qnn-context-binary-generator"
else
    echo "WARNING: qnn-context-binary-generator not found at $QNN_CONTEXT_GEN"
fi

PUSH_LIBS_END=$(date +%s.%N)
PUSH_LIBS_TIME=$(get_elapsed_ms $PUSH_LIBS_START $PUSH_LIBS_END)
echo "[OK] QNN libraries pushed (${PUSH_LIBS_TIME} ms)"

echo "--- Generating context binary on device ---"
CONTEXT_GEN_START=$(date +%s.%N)

# Run context binary generator synchronously (not in background)
# For large models (1M), this can take 30-60+ seconds
adb shell "cd $REMOTE_DIR && \
           rm -rf output model_context.bin && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qnn-context-binary-generator --model libmodel.so --backend libQnnHtp.so --binary_file model_context 2>&1" || true

# Wait and check for output
MAX_WAIT=120
WAITED=0
echo "Waiting for context binary generation (max ${MAX_WAIT}s)..."
while [ $WAITED -lt $MAX_WAIT ]; do
    if adb shell "[ -f $REMOTE_DIR/output/model_context.bin ]" 2>/dev/null; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done
echo ""

if adb shell "[ -f $REMOTE_DIR/output/model_context.bin ]" 2>/dev/null; then
    adb shell "cd $REMOTE_DIR && cp output/model_context.bin model_context.bin"
    CONTEXT_GEN_END=$(date +%s.%N)
    CONTEXT_GEN_TIME=$(get_elapsed_ms $CONTEXT_GEN_START $CONTEXT_GEN_END)
    echo "[OK] Context binary generated (${CONTEXT_GEN_TIME} ms)"
else
    CONTEXT_GEN_END=$(date +%s.%N)
    CONTEXT_GEN_TIME=$(get_elapsed_ms $CONTEXT_GEN_START $CONTEXT_GEN_END)
    echo "ERROR: Context binary generation failed after ${WAITED}s"
    echo "Attempting to use model.bin directly as fallback..."
    # Try using model.bin directly (works for some QNN versions)
    adb shell "cd $REMOTE_DIR && cp model.bin model_context.bin"
fi

echo "--- Running benchmark on device ---"
BENCHMARK_START=$(date +%s.%N)
adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qidk_rag_demo model_context.bin query.fvecs results libQnnHtp.so docs.fvecs $TOP_K"
RESULT=$?
BENCHMARK_END=$(date +%s.%N)
BENCHMARK_TIME=$(get_elapsed_ms $BENCHMARK_START $BENCHMARK_END)

if [ $RESULT -eq 0 ]; then
    echo "[OK] Benchmark execution successful! (${BENCHMARK_TIME} ms)"
    
    mkdir -p "$LOCAL_RESULTS_DIR"
    
    # Time: Pull results
    PULL_START=$(date +%s.%N)
    adb pull $REMOTE_RESULTS_DIR/results.txt "$LOCAL_RESULTS_DIR/results.txt" >/dev/null 2>&1
    adb pull $REMOTE_RESULTS_DIR/metrics.txt "$LOCAL_RESULTS_DIR/metrics.txt" >/dev/null 2>&1
    PULL_END=$(date +%s.%N)
    PULL_TIME=$(get_elapsed_ms $PULL_START $PULL_END)
    
    echo "[OK] Results saved to $LOCAL_RESULTS_DIR/results.txt"
    echo "[OK] Metrics saved to $LOCAL_RESULTS_DIR/metrics.txt"
    
    # Calculate total deploy time
    DEPLOY_END_TIME=$(date +%s.%N)
    TOTAL_DEPLOY_TIME=$(get_elapsed_ms $DEPLOY_START_TIME $DEPLOY_END_TIME)
    
    echo ""
    echo "=== Timing Breakdown ==="
    echo "  Push executable:     ${PUSH_EXE_TIME} ms"
    echo "  Push model binary:   ${PUSH_MODEL_TIME} ms"
    echo "  Push data binary:    ${PUSH_DATA_TIME} ms"
    echo "  Push query file:     ${PUSH_QUERY_TIME} ms"
    echo "  Push doc file:       ${PUSH_DOC_TIME} ms"
    echo "  Push QNN libs:       ${PUSH_LIBS_TIME} ms"
    echo "  Context generation:  ${CONTEXT_GEN_TIME} ms"
    echo "  Benchmark execution: ${BENCHMARK_TIME} ms"
    echo "  Pull results:        ${PULL_TIME} ms"
    echo "  ---------------------------------"
    echo "  TOTAL DEPLOY TIME:   ${TOTAL_DEPLOY_TIME} ms"
    echo ""
    
    # Save timing to a separate file
    TIMING_FILE="$LOCAL_RESULTS_DIR/timing.txt"
    cat > "$TIMING_FILE" << EOF
=== Deploy Timing Breakdown ===
Date: $(date)
Dataset: $DATASET_NAME
Model size: $MODEL_SIZE_SUFFIX
Batch size: $BATCH_SIZE

Push executable:     ${PUSH_EXE_TIME} ms
Push model binary:   ${PUSH_MODEL_TIME} ms
Push data binary:    ${PUSH_DATA_TIME} ms
Push query file:     ${PUSH_QUERY_TIME} ms
Push doc file:       ${PUSH_DOC_TIME} ms
Push QNN libs:       ${PUSH_LIBS_TIME} ms
Context generation:  ${CONTEXT_GEN_TIME} ms
Benchmark execution: ${BENCHMARK_TIME} ms
Pull results:        ${PULL_TIME} ms
---------------------------------
TOTAL DEPLOY TIME:   ${TOTAL_DEPLOY_TIME} ms
EOF
    echo "[OK] Timing saved to $TIMING_FILE"
    
    echo ""
    echo "=== Performance Summary ==="
    if [ -f "$LOCAL_RESULTS_DIR/metrics.txt" ]; then
        grep -E "Total execution time|Throughput|Average latency" "$LOCAL_RESULTS_DIR/metrics.txt" | head -10
    fi
    
    echo ""
    echo "Full metrics: $LOCAL_RESULTS_DIR/metrics.txt"
    echo "Full results: $LOCAL_RESULTS_DIR/results.txt"
else
    echo "[ERROR] Execution failed with code $RESULT (${BENCHMARK_TIME} ms)"
fi

exit 0