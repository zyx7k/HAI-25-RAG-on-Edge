#!/bin/bash
set -e

# --- CONFIGURATION ---
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"
USE_FULL_DATASET=false
# --- END CONFIGURATION ---

# Local paths
HOST_EXE=./android/output/qidk_rag_demo
if [ "$USE_FULL_DATASET" = true ]; then
    MODEL_BIN=./android/app/src/main/assets/vector_search_1M.bin
else
    MODEL_BIN=./android/app/src/main/assets/vector_search_10k.bin
fi
QUERY_FILE=./data/siftsmall_query.fvecs

# QNN libs
QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
BUILT_LIBS_DIR=./android/app/main/libs/arm64-v8a

QNN_LIBS=("libQnnHtp.so" "libQnnSystem.so" "libc++_shared.so")
OPTIONAL_LIBS=(
    "libQnnHtpNetRunExtensions.so"
    "libQnnHtpV73Stub.so"
    "libQnnHtpV75Stub.so"
    "libQnnHtpV79Stub.so"
)
ADSP_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
ADSP_LIBS=("libQnnHtpV73Skel.so" "libQnnHtpV75Skel.so" "libQnnHtpV79Skel.so")

# Remote paths
REMOTE_DIR=/data/local/tmp/qnn-rag-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo
REMOTE_MODEL=$REMOTE_DIR/$(basename $MODEL_BIN)
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_RESULTS=$REMOTE_DIR/results.txt

echo "=== Deploying QNN RAG Demo ==="

# Check device
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No device detected. Connect via ADB."
    exit 1
fi
echo "✓ Device connected"

# Validate files
for f in "$HOST_EXE" "$MODEL_BIN" "$QUERY_FILE"; do
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
adb push $MODEL_BIN $REMOTE_MODEL >/dev/null
adb push $QUERY_FILE $REMOTE_QUERY >/dev/null
echo "✓ Model and query data pushed"

# Step 4: Push core QNN libs
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        adb push "$BUILT_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
    else
        echo "⚠️  Missing $lib"
    fi
done

# Step 5: Push optional stub libs
for lib in "${OPTIONAL_LIBS[@]}"; do
    [ -f "$QNN_LIBS_DIR/$lib" ] && adb push "$QNN_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
done

# Step 6: Push DSP skeleton libs
SKEL_PUSHED=false
for lib in "${ADSP_LIBS[@]}"; do
    if [ -f "$ADSP_LIBS_DIR/$lib" ]; then
        adb push "$ADSP_LIBS_DIR/$lib" $REMOTE_DIR/ >/dev/null
        SKEL_PUSHED=true
    fi
done
[ "$SKEL_PUSHED" = false ] && echo "⚠️  No skeleton library found. Falling back to CPU."

# Step 7: Run inference
echo "--- Running on device ---"
adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=$REMOTE_DIR:\$ADSP_LIBRARY_PATH && \
           $REMOTE_EXE $REMOTE_MODEL $REMOTE_QUERY $REMOTE_RESULTS libQnnHtp.so"
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
