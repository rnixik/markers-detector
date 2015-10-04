LOCAL_PATH := $(call my-dir)

OPENCV_CAMERA_MODULES:=off
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=STATIC
include S:\Programs\NVPACK\OpenCV-2.4.8.2-Tegra-sdk\sdk\native\jni\OpenCV.mk
#include S:\Programs\OpenCV-3.0.0-android-sdk-1\OpenCV-android-sdk\sdk\native\jni\OpenCV.mk

LOCAL_MODULE    := markersdetector
LOCAL_CFLAGS    := -std=gnu++11
LOCAL_SRC_FILES := ../source/MarkersDetector.cpp 
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../source
LOCAL_LDLIBS    += -llog -ldl -lGLESv2
include $(BUILD_SHARED_LIBRARY)
