# Android.mk for ndk-build
# This file must be in the `jni` directory
LOCAL_PATH := $(call my-dir)

# --- Define libQnnSystem prebuilt module ---
include $(CLEAR_VARS)
LOCAL_MODULE := libQnnSystem
LOCAL_SRC_FILES := $(QNN_SDK_ROOT)/lib/aarch64-android/libQnnSystem.so
include $(PREBUILT_SHARED_LIBRARY)

# --- Define libQnnHtp prebuilt module (Hexagon backend) ---
include $(CLEAR_VARS)
LOCAL_MODULE := libQnnHtp
LOCAL_SRC_FILES := $(QNN_SDK_ROOT)/lib/aarch64-android/libQnnHtp.so
include $(PREBUILT_SHARED_LIBRARY)

# --- Build QIDK RAG Demo Executable ---
include $(CLEAR_VARS)
LOCAL_MODULE := qidk_rag_demo
LOCAL_MODULE_TAGS := optional

# Add source files
LOCAL_SRC_FILES := \
    main.cpp \
    QnnRunner.cpp

# Add C/C++ flags
LOCAL_CPPFLAGS += -std=c++17 -Wall -Werror -fPIE -fexceptions
LOCAL_CPP_FEATURES += exceptions rtti

# Add QNN SDK include paths
LOCAL_C_INCLUDES += \
    $(QNN_SDK_ROOT)/include/QNN \
    $(QNN_SDK_ROOT)/include/QNN/System

# Link flags for position independent executable
LOCAL_LDFLAGS += -fPIE -pie

# System libraries
LOCAL_LDLIBS += -ldl -llog

# Link against QNN libraries
LOCAL_SHARED_LIBRARIES := libQnnSystem libQnnHtp

include $(BUILD_EXECUTABLE)