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

# --- Brute Force Executable ---
include $(CLEAR_VARS)
LOCAL_MODULE := qidk_rag_demo
LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    main.cpp \
    QnnRunner.cpp

LOCAL_CPPFLAGS += -std=c++17 -Wall -Wno-unused-variable -Wno-unused-parameter -fPIE -fexceptions
LOCAL_CPP_FEATURES += exceptions rtti

LOCAL_C_INCLUDES += \
    $(QNN_SDK_ROOT)/include \
    $(QNN_SDK_ROOT)/include/QNN \
    $(LOCAL_PATH)

LOCAL_LDFLAGS += -fPIE -pie
LOCAL_LDLIBS += -ldl -llog -pthread
LOCAL_SHARED_LIBRARIES := libQnnSystem libQnnHtp

include $(BUILD_EXECUTABLE)

# --- IVF Search Executable ---
include $(CLEAR_VARS)
LOCAL_MODULE := qidk_ivf
LOCAL_MODULE_TAGS := optional

LOCAL_SRC_FILES := \
    main_ivf.cpp \
    QnnRunner.cpp \
    IVFIndex.cpp

LOCAL_CPPFLAGS += -std=c++17 -Wall -Wno-unused-variable -Wno-unused-parameter -fPIE -fexceptions -fopenmp -O3
LOCAL_CPP_FEATURES += exceptions rtti

LOCAL_C_INCLUDES += \
    $(QNN_SDK_ROOT)/include \
    $(QNN_SDK_ROOT)/include/QNN \
    $(LOCAL_PATH)

LOCAL_LDFLAGS += -fPIE -pie -fopenmp -static-openmp
LOCAL_LDLIBS += -ldl -llog -pthread
LOCAL_SHARED_LIBRARIES := libQnnSystem libQnnHtp

include $(BUILD_EXECUTABLE)
