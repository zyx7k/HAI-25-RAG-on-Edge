#!/bin/bash
set -e

DATASET=${1:-sift}
NPROBE=${2:-16}
REORDERED=${3:-}
DEVICE_DIR="/data/local/tmp/ivf_search"

# Determine index directory
if [ "$REORDERED" = "reordered" ]; then
    INDEX_DIR="models/ivf_${DATASET}_reordered"
    SUFFIX="_reordered"
else
    INDEX_DIR="models/ivf_${DATASET}"
    SUFFIX=""
fi

echo "Deploying IVF Search: ${DATASET}${SUFFIX}"

if [ ! -d "${INDEX_DIR}" ]; then
    echo "ERROR: Index directory not found: ${INDEX_DIR}"
    if [ "$REORDERED" = "reordered" ]; then
        echo "Run: python3 prepare/create_ivf_model_reordered.py ${DATASET}"
    else
        echo "Run: python3 prepare/create_ivf_model.py ${DATASET}"
    fi
    exit 1
fi

if [ ! -f "${INDEX_DIR}/centroids.bin" ]; then
    echo "Warning: centroids.bin not found"
fi

adb shell "mkdir -p ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}"

adb push android/app/main/libs/arm64-v8a/qidk_ivf ${DEVICE_DIR}/

adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}/
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}/ 2>/dev/null || true
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}/ 2>/dev/null || true

adb push ${INDEX_DIR}/ivf_config.json ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
adb push ${INDEX_DIR}/cluster_offsets.npy ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/

# Push appropriate files based on mode
if [ "$REORDERED" = "reordered" ]; then
    adb push ${INDEX_DIR}/vectors_reordered.npy ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
    adb push ${INDEX_DIR}/reorder_to_original.npy ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
else
    adb push ${INDEX_DIR}/cluster_indices.npy ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
    adb push ${INDEX_DIR}/vectors.npy ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
fi

if [ -f "${INDEX_DIR}/centroids.bin" ]; then
    adb push ${INDEX_DIR}/centroids.bin ${DEVICE_DIR}/ivf_${DATASET}${SUFFIX}/
fi

adb push data/${DATASET}/${DATASET}_query.fvecs ${DEVICE_DIR}/
adb push data/${DATASET}/${DATASET}_groundtruth.ivecs ${DEVICE_DIR}/

adb shell "chmod +x ${DEVICE_DIR}/qidk_ivf"
adb shell "chmod 755 ${DEVICE_DIR}/*.so"

echo "Deployment complete"
echo "Run: adb shell 'cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && ./qidk_ivf ./ivf_${DATASET}${SUFFIX} ${DATASET}_query.fvecs results ./libQnnHtp.so 10 ${NPROBE} ${DATASET}_groundtruth.ivecs'"
