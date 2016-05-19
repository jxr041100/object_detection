LOCAL_PATH := $(call my-dir)/../../CSIM
include $(CLEAR_VARS)

LOCAL_MODULE     := libkneron_adas


LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_SRC_FILES  += adas.cpp \
                    ObjDetectorC.cpp 

LOCAL_MODULE_OWNER := kneron
LOCAL_LDLIBS +=  -llog -ldl 

include $(BUILD_SHARED_LIBRARY)
