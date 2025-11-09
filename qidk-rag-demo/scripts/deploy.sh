#!/binin/bash
set -e

# --- CONFIGURE THESE PATHS ---
# Assumes QAIED SDK is in ~/qualcomm/qai-direct-sdk
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
# These are the backend implementations
QNN_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android
QNN_LIBS=(
    "libQnnHtp.so"
    "libQnnHtpNetRunExtensions.so"
    "libQnnHtpV73Stub.so" # This may change based on your chip, V73 is for SD 8 Gen 2
    "libQnnSystem.so"
    "libQnnManager.so" # The app is linked against this
)
# This is the DSP implementation
ADSP_LIBS_DIR=$QNN_SDK_ROOT/lib/aarch64-android/unsigned/
ADSP_LIB="libQnnHtpV73Skel.so" # This is the library for the DSP

# On-device paths
REMOTE_DIR=/data/local/tmp/qnn-rag-demo
REMOTE_EXE=$REMOTE_DIR/qidk_rag_demo
REMOTE_MODEL=$REMOTE_DIR/$(basename $MODEL_BIN)
REMOTE_QUERY=$REMOTE_DIR/query.fvecs
REMOTE_RESULTS=$REMOTE_DIR/results.txt

echo "--- Deploying to device ---"

# 1. Create dir and push files
adb shell "mkdir -p $REMOTE_DIR"
adb push $HOST_EXE $REMOTE_EXE
adb push $MODEL_BIN $REMOTE_MODEL
adb push $QUERY_FILE $REMOTE_QUERY

# 2. Push QNN libraries
echo "Pushing QNN libraries..."
for lib in "${QNN_LIBS[@]}"; do
    adb push $QNN_LIBS_DIR/$lib $REMOTE_DIR/
done

echo "Pushing ADSP (DSP) library..."
adb push $ADSP_LIBS_DIR/$ADSP_LIB $REMOTE_DIR/

# 3. Run the command
echo "--- Running executable on device ---"
# This is the most important part
# LD_LIBRARY_PATH tells the executable where to find libQnnManager.so
# ADSP_LIBRARY_PATH tells the QNN system where to find the DSP skeleton lib
adb shell "export LD_LIBRARY_PATH=$REMOTE_DIR && \
           export ADSP_LIBRARY_PATH=$REMOTE_DIR && \
           chmod +x $REMOTE_EXE && \
           $REMOTE_EXE $REMOTE_MODEL $REMOTE_QUERY $REMOTE_RESULTS $REMOTE_DIR/libQnnHtp.so"

echo "--- Run complete ---"

# 4. Pull results
adb pull $REMOTE_RESULTS ./results.txt
echo "Results pulled to ./results.txt"

# 5. Clean up (optional)
# adb shell "rm -rf $REMOTE_DIR"
# echo "Cleaned up remote directory."

echo "Done."