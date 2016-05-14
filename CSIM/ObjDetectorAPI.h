#ifndef OBJ_DETECTOR_API_H
#define OBJ_DETECTOR_API_H
#include "adas.h"

typedef struct
{
    float x;
    float y;
}Point2Df;

typedef struct
{
    uint32_t topx;
    uint32_t topy;
    uint32_t width;
    uint32_t height;
}icvRect;

typedef struct 
{
    /// flag to inidate whether the bounding box will be used as mask for face detection
    uint32_t ROIflag;
    /// top left x coordinate of the bounding box
    uint32_t topx;
    /// top left y coordinate of the bounding box
    uint32_t topy;
    /// width of the bounding box
    uint32_t width;
    /// height of the bounding box
    uint32_t height;
    /// face ID for tracking
    uint32_t ID;
}cvFacedetectResult;


typedef struct
{
     uint32_t minSize;
     uint32_t maxSize;
     uint32_t stepSize; 
    float scaleFactor; 


}cvFacedetectParameters;


bool face_detection_process( uint8_t * __restrict src, 
      uint32_t srcWidth, 
      uint32_t srcHeight, 
      uint32_t srcStride,
      cvFacedetectParameters *para, 
      uint32_t maxDetectedFaceNum,
      uint32_t *resultFaceNum,
      cvFacedetectResult *result
);


bool init_face_detection(uint32_t Height,uint32_t Width,cvFacedetectParameters *params);

void deinit_face_detection();
#endif
