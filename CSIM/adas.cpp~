#include "adas.h"
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include "ObjDetectorAPI.h"

#include <iostream>
#include <fstream>

 cvFacedetectParameters detParam; 
 int _height;
 int _width;


int adas_car_detect(uint8_t *frame, int width, int height, CarDistance *distance, uint32_t *count)
{
	 cvFacedetectResult result[25];  
	
	 int ret = face_detection_process(frame, width, height, width, &detParam,  20, count, result);
	 //int N = (int)*count
	 for( uint32_t i = 0 ; i < *count; i++)
	 {
		 distance[i].x = result[i].topx;
		 distance[i].y = result[i].topy;
		 distance[i].width = result[i].width;
		 distance[i].height = result[i].height;

		 distance[i].distance = result[i].ID;
	 }
	 return 0;
}

int adas_car_detect(uint8_t *frame, CarDistance *distance, uint32_t *count)
{
	return adas_car_detect(frame, _width, _height, distance, count);
}

void adas_free()
{
	deinit_face_detection();
}



int adas_init(int width, int height)
{
	bool ret = init_face_detection(height, width, &detParam);
	_width = width;
	_height = height;
	if(ret == false)
	{
		return -1;
	} else {
		return 0;
	}
}
