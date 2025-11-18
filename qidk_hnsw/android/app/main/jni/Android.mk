LOCAL_PATH := $(call my-dir)

# --- Prebuilt QNN libraries ---
include $(CLEAR_VARS)
LOCAL_MODULE := libQnnSystem
LOCAL_SRC_FILES := $(QNN_SDK_ROOT)/lib/aarch64-android/libQnnSystem.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libQnnHtp
LOCAL_SRC_FILES := $(QNN_SDK_ROOT)/lib/aarch64-android/libQnnHtp.so
include $(PREBUILT_SHARED_LIBRARY)

# --- Executable target ---
include $(CLEAR_VARS)
LOCAL_MODULE := qidk_rag_demo
LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    main.cpp \
    hnsw_search.cpp \
    QnnRunner.cpp

LOCAL_CPPFLAGS += -std=c++17 -Wall -Wno-unused-variable -Wno-unused-parameter -fPIE -fexceptions
LOCAL_CPP_FEATURES += exceptions rtti

# Include QNN headers and local sources
LOCAL_C_INCLUDES += \
    $(QNN_SDK_ROOT)/include \
    $(QNN_SDK_ROOT)/include/QNN \
    $(LOCAL_PATH)

LOCAL_LDFLAGS += -fPIE -pie
LOCAL_LDLIBS += -ldl -llog
LOCAL_SHARED_LIBRARIES := libQnnSystem libQnnHtp

include $(BUILD_EXECUTABLE)