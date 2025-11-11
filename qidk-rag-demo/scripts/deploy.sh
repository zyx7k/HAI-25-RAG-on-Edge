#!/bin/bash
set -e

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"
USE_FULL_DATASET=false

HOST_EXE=./android/output/qidk_rag_demo
MODEL_BIN=./qnn/qnn_artifacts/model.bin
if [ "$USE_FULL_DATASET" = true ]; then
    DATA_BIN=./android/app/src/main/assets/vector_search_1M.bin
else
    DATA_BIN=./android/app/src/main/assets/vector_search_10k.bin
fi
QUERY_FILE=./data/siftsmall_query.fvecs
DOC_FILE=./data/siftsmall_base.fvecs

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
REMOTE_RESULTS=$REMOTE_DIR/results.txt

echo "=== Deploying QNN RAG Demo ==="

if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device detected. Connect via ADB."
    exit 1
fi
echo "✓ Device connected"

for f in "$HOST_EXE" "$MODEL_BIN" "$DATA_BIN" "$QUERY_FILE" "$DOC_FILE"; do
    [ ! -f "$f" ] && echo "ERROR: Missing $f" && exit 1
done

adb shell "mkdir -p $REMOTE_DIR"
echo "✓ Created $REMOTE_DIR"

adb push $HOST_EXE $REMOTE_EXE >/dev/null
adb shell "chmod +x $REMOTE_EXE"
echo "✓ Executable pushed"

adb push $MODEL_BIN $REMOTE_MODEL_BIN >/dev/null
adb push $DATA_BIN $REMOTE_DATA >/dev/null
adb push $QUERY_FILE $REMOTE_QUERY >/dev/null
adb push $DOC_FILE $REMOTE_DOC >/dev/null
echo "✓ Model binary, query data, and document embeddings pushed"

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
            echo "✓ Pushed $skel"
            SKEL_PUSHED=true
        fi
    done
done

if [ "$SKEL_PUSHED" = false ]; then
    echo "WARNING: No Hexagon skel libraries found."
fi

adb push ./qnn/qnn_artifacts/aarch64-android/libmodel.so $REMOTE_DIR/ >/dev/null

echo "--- Generating context binary on device ---"
adb shell "cd $REMOTE_DIR && \
           rm -rf output model_context.bin && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           ./qnn-context-binary-generator --model libmodel.so --backend libQnnHtp.so --binary_file model_context 2>&1 | grep 'pid:' &"

sleep 2

if adb shell "[ -f $REMOTE_DIR/output/model_context.bin ]"; then
    adb shell "cd $REMOTE_DIR && cp output/model_context.bin model_context.bin"
    echo "✓ Context binary generated"
else
    echo "⚠️  Context binary generation may have failed"
fi

echo "--- Running on device ---"
adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qidk_rag_demo model_context.bin query.fvecs results.txt libQnnHtp.so docs.fvecs"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "✓ Execution successful!"
    
    mkdir -p results
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="results/knn_results_${TIMESTAMP}.txt"
    
    adb pull $REMOTE_RESULTS $RESULT_FILE >/dev/null 2>&1
    echo "✓ Results saved to $RESULT_FILE"
    
    echo "--- Results (first 20 lines) ---"
    head -20 $RESULT_FILE
    
    echo ""
    echo "Full results: $RESULT_FILE"
else
    echo "✗ Execution failed with code $RESULT"
fi
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device detected. Connect via ADB."
    exit 1
fi
echo "✓ Device connected"

# Validate files
for f in "$HOST_EXE" "$MODEL_BIN" "$DATA_BIN" "$QUERY_FILE" "$DOC_FILE"; do
    [ ! -f "$f" ] && echo "ERROR: Missing $f" && exit 1
done

# Step 1: Create remote directory
adb shell "mkdir -p $REMOTE_DIR"
echo "✓ Created $REMOTE_DIR"

# Step 2: Push executable
adb push $HOST_EXE $REMOTE_EXE >/dev/null
adb shell "chmod +x $REMOTE_EXE"
echo "✓ Executable pushed"

# Step 3: Push model + data
adb push $MODEL_BIN $REMOTE_MODEL_BIN >/dev/null
adb push $DATA_BIN $REMOTE_DATA >/dev/null
adb push $QUERY_FILE $REMOTE_QUERY >/dev/null
adb push $DOC_FILE $REMOTE_DOC >/dev/null
echo "✓ Model binary, query data, and document embeddings pushed"

# Step 4: Push core QNN libs
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        adb push "$BUILT_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    elif [ -f "$QNN_LIBS_DIR/$lib" ]; then
        adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    else
        echo "⚠️  Missing $lib"
    fi
done

# Step 5: Push optional stub libs
for lib in "${OPTIONAL_LIBS[@]}"; do
    [ -f "$QNN_LIBS_DIR/$lib" ] && adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
done

# Step 6: Push DSP skeleton libs
echo "Searching for NPU/Skel libs..."
SKEL_PUSHED=false
# *** FIXED: The paths in this array are now correct ***
HEXAGON_DIRS=(
    "$QNN_SDK_ROOT/lib/hexagon-v73/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned"
    "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned"
)
ADSP_LIBS=("libQnnHtpV73Skel.so" "libQnnHtpV75Skel.so" "libQnnHtpV79Skel.so")

for dir in "${HEXAGON_DIRS[@]}"; do
    for lib in "${ADSP_LIBS[@]}"; do
        if [ -f "$dir/$lib" ]; then
            adb push "$dir/$lib" $REMOTE_DIR/ >/dev/null
            echo "✓ Pushed $lib"
            SKEL_PUSHED=true
        fi
    done
done
[ "$SKEL_PUSHED" = false ] && echo "⚠️  No skeleton library found. NPU may fail."

# Step 7: Generate context binary on device (with HtpPrepare lib)
echo "--- Generating context binary on device ---"
adb shell "cd $REMOTE_DIR && \
    rm -rf output model_context.bin && \
    export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=. && \
    ./qnn-context-binary-generator --model libmodel.so --backend libQnnHtp.so --binary_file model_context && \
    cp output/model_context.bin model_context.bin 2>/dev/null || true"

if adb shell "[ -f $REMOTE_DIR/model_context.bin ] && echo exists" | grep -q exists; then
    echo "✓ Context binary generated"
else
    echo "⚠️  Context binary generation may have failed"
fi

# Step 8: Run inference
echo "--- Running on device ---"
adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=. && \
           ./qidk_rag_demo model_context.bin query.fvecs results.txt libQnnHtp.so docs.fvecs"
RESULT=$?

# Step 8: Results
if [ $RESULT -eq 0 ]; then
    echo "✓ Execution successful!"
    adb pull $REMOTE_RESULTS ./results.txt >/dev/null
    echo "--- Results (first 20 lines) ---"
    head -20 ./results.txt
    echo ""
    echo "Full results: ./results.txt"
    echo "Remote dir: $REMOTE_DIR"
    echo "Clean up: adb shell rm -rf $REMOTE_DIR"
else
    echo "❌ Execution failed (code $RESULT)"
    echo "Debug:"
    echo "  adb logcat | grep -i qnn"
    echo "  Try CPU backend: libQnnCpu.so"
    echo "  adb shell ls -la $REMOTE_DIR"
    exit 1
fi