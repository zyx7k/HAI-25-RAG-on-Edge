#!/bin/bash
set -e

# --- CONFIGURE THESE PATHS ---
: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

# Set to true/false to match build.sh
USE_FULL_DATASET=false

# --- END CONFIGURATION ---

# Local file paths
HOST_EXE=./android/output/qidk_rag_demo
if [ "$USE_FULL_DATASET" = true ]; then
    MODEL_BIN=./android/app/src/main/assets/vector_search_1M.bin
    QUERY_FILE=./data/siftsmall_query.fvecs
else
    MODEL_BIN=./android/app/src/main/assets/vector_search_10k.bin
    QUERY_FILE=./data/siftsmall_query.fvecs
fi

# QNN Libraries to push
QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android

# NOTE: We're using the libraries that were actually built and installed by ndk-build
# These are in android/app/main/libs/arm64-v8a/
BUILT_LIBS_DIR=./android/app/main/libs/arm64-v8a

QNN_LIBS=(
    "libQnnHtp.so"
    "libQnnSystem.so"
    "libc++_shared.so"
)

# Additional libraries that might be needed (check if they exist)
OPTIONAL_LIBS=(
    "libQnnHtpNetRunExtensions.so"
    "libQnnHtpV73Stub.so"  # For Snapdragon 8 Gen 2
    "libQnnHtpV75Stub.so"  # For Snapdragon 8 Gen 3
    "libQnnHtpV79Stub.so"  # For Snapdragon 8 Gen 4/Elite
)

# DSP Skeleton library (device-specific)
ADSP_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android/
ADSP_LIBS=(
    "libQnnHtpV73Skel.so"  # SD 8 Gen 2
    "libQnnHtpV75Skel.so"  # SD 8 Gen 3  
    "libQnnHtpV79Skel.so"  # SD 8 Gen 4/Elite
)

# On-device paths
REMOTE_DIR=/data/local/tmp/qnn-rag-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo
REMOTE_MODEL=$REMOTE_DIR/$(basename $MODEL_BIN)
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_RESULTS=$REMOTE_DIR/results.txt

echo "=== Deploying QNN RAG Demo to Device ==="
echo ""

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No Android device detected. Please connect via ADB."
    echo "Run 'adb devices' to check connection."
    exit 1
fi

echo "✓ Device connected"
echo ""

# Check if files exist
if [ ! -f "$HOST_EXE" ]; then
    echo "ERROR: Executable not found at $HOST_EXE"
    echo "Please run build.sh first."
    exit 1
fi

if [ ! -f "$MODEL_BIN" ]; then
    echo "ERROR: Model binary not found at $MODEL_BIN"
    exit 1
fi

if [ ! -f "$QUERY_FILE" ]; then
    echo "ERROR: Query file not found at $QUERY_FILE"
    exit 1
fi

echo "--- Step 1: Creating directory on device ---"
adb shell "mkdir -p $REMOTE_DIR"
echo "✓ Created $REMOTE_DIR"

echo ""
echo "--- Step 2: Pushing executable ---"
adb push $HOST_EXE $REMOTE_EXE
adb shell "chmod +x $REMOTE_EXE"
echo "✓ Pushed and made executable"

echo ""
echo "--- Step 3: Pushing model and data ---"
adb push $MODEL_BIN $REMOTE_MODEL
adb push $QUERY_FILE $REMOTE_QUERY
echo "✓ Model and query data pushed"

echo ""
echo "--- Step 4: Pushing QNN libraries (from build output) ---"
for lib in "${QNN_LIBS[@]}"; do
    if [ -f "$BUILT_LIBS_DIR/$lib" ]; then
        echo "  Pushing $lib..."
        adb push $BUILT_LIBS_DIR/$lib $REMOTE_DIR/
    else
        echo "  WARNING: $lib not found in $BUILT_LIBS_DIR"
    fi
done

echo ""
echo "--- Step 5: Pushing optional QNN stub libraries ---"
for lib in "${OPTIONAL_LIBS[@]}"; do
    if [ -f "$QNN_LIBS_DIR/$lib" ]; then
        echo "  Pushing $lib..."
        adb push $QNN_LIBS_DIR/$lib $REMOTE_DIR/
    else
        echo "  Skipping $lib (not found)"
    fi
done

echo ""
echo "--- Step 6: Pushing DSP Skeleton library ---"
SKEL_PUSHED=false
for lib in "${ADSP_LIBS[@]}"; do
    if [ -f "$ADSP_LIBS_DIR/$lib" ]; then
        echo "  Pushing $lib..."
        adb push $ADSP_LIBS_DIR/$lib $REMOTE_DIR/
        SKEL_PUSHED=true
    fi
done

if [ "$SKEL_PUSHED" = false ]; then
    echo "  WARNING: No skeleton library found. HTP backend might not work."
    echo "  Device will fall back to CPU backend."
fi

echo ""
echo "--- Step 7: Running inference on device ---"
echo ""
echo "Command: $REMOTE_EXE $REMOTE_MODEL $REMOTE_QUERY $REMOTE_RESULTS $REMOTE_DIR/libQnnHtp.so"
echo ""

# Run with proper environment variables
adb shell "cd $REMOTE_DIR && \
           export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=$REMOTE_DIR:\$ADSP_LIBRARY_PATH && \
           $REMOTE_EXE $REMOTE_MODEL $REMOTE_QUERY $REMOTE_RESULTS libQnnHtp.so"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✓ Execution completed successfully!"
    
    echo ""
    echo "--- Step 8: Pulling results ---"
    adb pull $REMOTE_RESULTS ./results.txt
    
    echo ""
    echo "=== Results Preview ==="
    echo ""
    head -20 ./results.txt
    
    echo ""
    echo "=== Summary ==="
    echo "✓ Full results saved to: ./results.txt"
    echo "✓ Remote directory: $REMOTE_DIR"
    echo ""
    echo "To view all results: cat ./results.txt"
    echo "To clean up device: adb shell rm -rf $REMOTE_DIR"
else
    echo ""
    echo "ERROR: Execution failed with code $RESULT"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check device logs: adb logcat | grep -i qnn"
    echo "2. Try CPU backend: Edit script to use libQnnCpu.so instead"
    echo "3. Verify device supports HTP: adb shell getprop ro.product.model"
    echo "4. Check remote directory: adb shell ls -la $REMOTE_DIR"
    exit 1
fi