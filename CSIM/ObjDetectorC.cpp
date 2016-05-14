#include <math.h>
#include <errno.h>
#include "ObjDetectorC.h"
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <pthread.h>
using namespace std;
void *buffer;
typedef struct _stripRange {
	ifcvFeature *feature;
    int start;
    int end;
    float sFactor;
	uint32_t processingRectSizeWidth;
	uint32_t sumStride;
	int stepSize;
    uint32_t windowSizeWidth, windowSizeHeight;
} Range;

pthread_mutex_t mutex;
icvRect *candidateResult = NULL;
uint32_t nCandidate;

bool init_face_detection(uint32_t Height,uint32_t Width,cvFacedetectParameters *param)
{   
    param->maxSize = 400;
    param->minSize = 30;
    param->scaleFactor = 1.2;
    param->stepSize = 2;	

    buffer = NULL;
    int maxDetectedFaceNum = 50;
    int memSize = (4 * Width * Height) + ((Width+9) * (Height+1)* 2) + (4788*2*4+200) + maxDetectedFaceNum * 100 * 16;

	buffer = malloc( (memSize * sizeof(uint8_t)));
    if(buffer!=NULL)        	
        memset(buffer,0,(memSize * sizeof(uint8_t)));
     else    
        return false;	

   pthread_mutex_init(&mutex, NULL);
   nCandidate =0;


    return true;
}

bool face_detection_process(uint8_t * __restrict src,                       
                            uint32_t srcWidth, 
                            uint32_t srcHeight, 
                            uint32_t srcStride,
                            cvFacedetectParameters *para,
                            uint32_t maxDetectedFaceNum,
                            uint32_t *resultFaceNum,
                            cvFacedetectResult *result)

{
    uint32_t r=1;
    *resultFaceNum=0;
    r=icvFaceDetection(src,srcWidth, srcHeight, srcStride, para, maxDetectedFaceNum, resultFaceNum, result, buffer);      
	return r>0?true:false;
}

void deinit_face_detection()
{
    pthread_mutex_destroy(&mutex);
	free(buffer);
}

int32_t absdef(int32_t a)
{
    if(a>=0)
        return a;
    else
        return -a;
}

int32_t icvSimilarRect(icvRect *r1, icvRect *r2)
{
    float d;    
    if(r1->width<=r2->width)
        d=(float)r1->width;
    else 
        d=(float)r2->width;
    if(r1->height<=r2->height)
        d+=r1->height;
    else 
        d+=r2->height;
    d=d*0.1f;
    int32_t result= (absdef(r1->topx-r2->topx)<=d && absdef(r1->topy-r2->topy)<=d &&
        absdef(r1->topx+r1->width-r2->topx-r2->width)<=d &&
        absdef(r1->topy+r1->height-r2->topy-r2->height)<=d);       
    return result;

}

int32_t icvMergePartition(icvRect * rectList, uint32_t rectNum, uint32_t *buff)
{

    uint32_t i,j,k; 
    uint32_t *buf=buff;
    uint32_t *p=buff+rectNum; //first part to save label
    uint32_t *computeMark=p+rectNum;
    icvRect *r1,*r2;

    uint32_t computemarkSum=0;
    uint32_t classIdx=0;
    uint32_t currentIdx=0;
    for(i=0;i<rectNum;i++)
    {
        buf[i]=0;
        p[i]=0;
        computeMark[i]=0;
    }
    while(computemarkSum<rectNum)
    {
        computeMark[currentIdx]=1;
        computemarkSum++;
        p[currentIdx]=1;
        r1=rectList+currentIdx;
        int totalNum=0;
        for(i=0;i<rectNum;i++)
        {
            if(computeMark[i]==0)
            {
                r2=(rectList+i);
                k=icvSimilarRect(r1,r2);  
                p[i]=k; 
                totalNum+=k;
            }    
        }


        while(totalNum>0)
        {
            for(i=0;i<rectNum;i++)
            {
                if(p[i]==1 && computeMark[i]==0)
                {
                    computeMark[i]=1;
                    computemarkSum++;                            
                    p[currentIdx]=1;
                    r1=rectList+i;
                    totalNum--;
                    for(j=0;j<rectNum;j++)
                    {
                        if(computeMark[j]==0 && p[j]==0)
                        {
                            r2=(rectList+j);
                            k=icvSimilarRect(r1,r2);  
                            p[j]=k; 
                            totalNum+=k;
                        }    
                    }
                }    
            }
        }

        while(computeMark[currentIdx]==1)
            currentIdx++;

        for(i=0;i<rectNum;i++)
        {
            if(p[i]==1)
            {
                buf[i]=classIdx;  
                p[i]=0;
            }      
        }
        classIdx++;              

    }

    return classIdx; 
}

uint32_t icvGrouprectangular(icvRect *rectList, uint32_t rectNum, uint32_t groupThreshold, icvRect *groupResult)
{


    int32_t resultFaceN=0;    
    if( groupThreshold <= 0 || rectNum==0 )
    {    
        return resultFaceN;             
    }

    void *buf= rectList+rectNum;

    uint8_t *ripHistogram = (uint8_t*)buf;
    memset(ripHistogram, 0, 12 * rectNum);
    buf = (uint8_t*)buf + 12 * rectNum;

    uint32_t *labels=(uint32_t *)buf;
    uint32_t nRecClass = icvMergePartition(rectList, rectNum, labels); //start from 1

    uint32_t i,j;    
    uint32_t *rectMergNum =(uint32_t *)buf+rectNum; 
    icvRect *rectClass = (icvRect *)(rectMergNum+rectNum);

    for(i=0;i<nRecClass;i++)
    {
        rectMergNum[i]=0;
        rectClass[i].topx=0;
        rectClass[i].topy=0;
        rectClass[i].width=0;
        rectClass[i].height=0;
    }

    for(i=0;i<rectNum;i++)
    {
        int cnt=labels[i];
        rectClass[cnt].topx += rectList[i].topx;
        rectClass[cnt].topy += rectList[i].topy;
        rectClass[cnt].width += rectList[i].width;
        rectClass[cnt].height += rectList[i].height;
        rectMergNum[cnt]++;

    } 
    for( i = 0; i < nRecClass; i++ )
    {
        float s = 1.f/rectMergNum[i];
        rectClass[i].topx = uint32_t((rectClass[i].topx*s) + 0.5);
        rectClass[i].topy = uint32_t((rectClass[i].topy*s) + 0.5);
        rectClass[i].width = uint32_t((rectClass[i].width*s) + 0.5);
        rectClass[i].height = uint32_t((rectClass[i].height*s) + 0.5);  
    }
    icvRect *r1,*r2;
    for( i = 0; i < nRecClass; i++ )
    {
        r1= (rectClass+i);
        uint32_t Num1 = rectMergNum[i];
        if( Num1 <= groupThreshold )
            continue;
        uint32_t M=3; 
        if (Num1>3)
            M=Num1;

        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nRecClass; j++ )
        {
            r2= (rectClass+j);
            uint32_t Num2 = rectMergNum[j];            
            if( j == i || Num2 <= groupThreshold )
                continue;  

            uint32_t dx = uint32_t(( r2->width * 0.2) + 0.5);
            uint32_t dy = uint32_t(( r2->height * 0.2) + 0.5);

            if( i != j &&
                r1->topx >= r2->topx - dx &&
                r1->topy >= r2->topy - dy &&
                r1->topx + rectClass[i].width <= r2->topx + r2->width + dx &&
                r1->topy + rectClass[i].height <= r2->topy + r2->height + dy) //&&
          
                break;
        }

        if( j == nRecClass )
        {
            groupResult[resultFaceN].topx=rectClass[i].topx;
            groupResult[resultFaceN].topy=rectClass[i].topy;
            groupResult[resultFaceN].width=rectClass[i].width;
            groupResult[resultFaceN].height=rectClass[i].height;
            resultFaceN++;
        }
    }
    return resultFaceN;
}


uint32_t  QuicksortPartition32UwIdx(uint32_t *src, uint32_t *idx, int32_t left, int32_t right, int32_t pivot)
{
    uint32_t *pointer;
    uint32_t pivotValue;
    uint32_t tempValue;
    uint32_t tempIdx;
    int32_t i;
    int32_t storeIndex;
    storeIndex = left;
    pointer = (uint32_t *)src;
    pivotValue = pointer[pivot];

    for (i = left; i < right; i ++) {
        if (pointer[i] > pivotValue) {
            tempValue = pointer[i];
            pointer[i] = pointer[storeIndex];
            pointer[storeIndex] = tempValue;
            tempIdx=idx[i];
            idx[i]=idx[storeIndex];
            idx[storeIndex]=tempIdx;
            storeIndex ++;
        }
    }

    tempValue = pointer[storeIndex];
    pointer[storeIndex] = pointer[right];
    pointer[right] = tempValue;

    tempIdx=idx[storeIndex];
    idx[storeIndex]=idx[right];
    idx[right]=tempIdx;

    return storeIndex;
}

void Quicksort32UwIdx(uint32_t *src, uint32_t *idx, int32_t left, int32_t right)
{
    int32_t pivotIndexValue;
    int32_t pivotIndexValueNew;

    if (left < right) {
        pivotIndexValue = right;
        pivotIndexValueNew = QuicksortPartition32UwIdx(src, idx,left, right, pivotIndexValue);
        Quicksort32UwIdx(src, idx, left, pivotIndexValueNew - 1);
        Quicksort32UwIdx(src, idx, pivotIndexValueNew + 1, right);
    }
}


uint8_t  QuicksortwIdx(uint32_t *src, uint32_t *idx, uint32_t numItems)
{   
    if (numItems <= 1) {
        return 0;
    }
    Quicksort32UwIdx(src, idx, 0, (numItems - 1));
    return 0;
}

void icvPickFace(uint32_t detFaceNum, uint32_t resultFaceNum, icvRect *buff, cvFacedetectResult *result)
{
    uint32_t widArr[1500]; //because can't more than 1500 faces in one image    
    uint32_t dstwidIdx[1500];

    uint32_t i,j;
    for(i=0;i<detFaceNum;i++){
        widArr[i]=buff[i].width*buff[i].height;
        dstwidIdx[i]=i;
    }        
    QuicksortwIdx(widArr, dstwidIdx,detFaceNum);

    //pick resultFaceNum face
    for(i=0;i<resultFaceNum;i++){
        j=dstwidIdx[i];
        result[i].topx=buff[j].topx;
        result[i].topy=buff[j].topy;
        result[i].width=buff[j].width;
        result[i].height=buff[j].height;
    }
}

double s1 =0;
double s2 =0;
double s3 =0;
double s4 =0;
double icvCounter =0;

int32_t icvRunFeature( const ifcvFeature *feature, uint32_t offset, int32_t * confidance, int sumStride, uint32_t x,uint16_t num_n_stages)
{

	icvCounter ++;
    int32_t nstages = num_n_stages;//MODEL_N_STAGES;
    int32_t stage;
    int32_t nodeOffset = 0, leafOffset = 0;
    uint32_t internalNodeSize = FACE_MODEL_N_INTRENAL_NODE;
    uint32_t ntrees, week, val; 
    int32_t  index, root;
    int32_t *internalNode;
    ifcvFeature ff;
    uint16_t test[21];
    uint16_t centerVal;
    uint32_t nWeak, jump, w;
    nWeak = 0;
    const uint16_t *p[16];
    uint32_t pBias;

	
    for(stage = 0; stage < nstages; stage++ )
    {
    
        ntrees = nStageWeekClassifierCount_[stage];
        *confidance = 0;   
        for( week = 0; week < ntrees; week++ )
        {    
   
            ff = feature[nodeOffset];
            w  = featureRectW[nodeOffset];   	
            p[0] = ff.p[0];
            p[1] = p[0]+w;
            p[2] = p[1]+w;
            p[3] = p[2]+w;
            p[4] = ff.p[1]; 
            p[5] = p[4]+w;
            p[6]= p[5]+w;
            p[7]= p[6]+w;
            pBias = p[4] - p[0];
            p[8]= p[4] + pBias;
            p[9]= p[8]+w;
            p[10]= p[9]+w;
            p[11]= p[10]+w;
            p[12]= p[8]+ pBias;
            p[13]= p[12]+w;
            p[14]= p[13]+w;
            p[15]= p[14]+w;
            test[0] = p[1][offset] - p[0][offset];
            test[1] = p[5][offset] - p[4][offset];
            test[2] = p[9][offset] - p[8][offset];
            test[3] = p[13][offset] - p[12][offset];
            test[4] = p[2][offset] - p[1][offset];
            test[5] = p[6][offset] - p[5][offset];
            test[6] = p[10][offset] - p[9][offset];
            test[7] = p[14][offset] - p[13][offset];
            test[8]  = p[3][offset] - p[2][offset];
            test[9]  = p[7][offset] - p[6][offset];
            test[10] = p[11][offset] - p[10][offset];
            test[11] = p[15][offset] - p[14][offset];
            centerVal = test[6] - test[5]-1;
            test[12] = test[1] - test[0];
            test[13] = test[5] - test[4];
            test[14] = test[9] - test[8] ;
            test[15] = test[10] - test[9];
            test[16] = test[11] - test[10];
            test[17] = test[7] - test[6];
            test[18] = test[3] - test[2];
            test[19] = test[2] - test[1];                 
            val =   (test[12] > centerVal ? 128 : 0) |   
                (test[13] > centerVal ? 64 : 0) |    
                (test[14] > centerVal ? 32 : 0) |    
                (test[15] > centerVal ? 16 : 0) |  
                (test[16] > centerVal ? 8 : 0)|  
                (test[17] > centerVal ? 4 : 0)|   
                (test[18] > centerVal ? 2 : 0)|    
                (test[19] > centerVal ? 1 : 0);                        
            internalNode = &cascadeInternalNodeValues_[nodeOffset*internalNodeSize]; 
            *confidance += cascadeLeaves_[internalNode[val>>5] & (1 << (val & 31)) ? leafOffset : leafOffset+1];        
            nodeOffset += 1;
            leafOffset += 2;
        }
        if( *confidance < stageThreshold[stage] )
        {
            return -stage;
		}          
    }
    return 1;
}

void *processStrip(void *_range)
{  
   Range *myRange = (Range*)_range;
   icvRect cf;
   int32_t confidance;

	for(int y = myRange->start; y < myRange->end; y +=  myRange->stepSize)
	{
		for(int x = 0; x < myRange->processingRectSizeWidth; x +=  myRange->stepSize )
		{
		    int offset = y * ( myRange->sumStride) + x;
		    int resultRunFearure = icvRunFeature( myRange->feature, offset, &confidance, myRange->sumStride,x);	
		    if( resultRunFearure > 0 ){
		        cf.topx = uint32_t((x* myRange->sFactor) + 0.5);
		        cf.topy = uint32_t((y* myRange->sFactor) + 0.5);
		        cf.width = myRange->windowSizeWidth;
		        cf.height = myRange->windowSizeHeight;
				pthread_mutex_lock(&mutex);
		        candidateResult[nCandidate++] = cf;
				pthread_mutex_unlock(&mutex);
		    } else if (resultRunFearure == 0) {
				x += myRange->stepSize;
			}
		}
	}
    cout << "exit the thread for range " << myRange->start << " and " << myRange->end << endl;
   //pthread_exit(NULL);
   return myRange;
}


uint32_t icvFaceDetection(  uint8_t * __restrict src,                          
                          uint32_t srcWidth, 
                          uint32_t srcHeight, 
                          uint32_t srcStride,
                          cvFacedetectParameters *para,
                          uint32_t maxDetectedFaceNum,
                          uint32_t *resultFaceNum,
                          cvFacedetectResult *result,
                          void *buffer)
{

    uint32_t x, y, fi, offset, minFaceSize, maxFaceSizeWidth, maxFaceSizeHeight, fX, fY, fW, fH, stepSize;
    uint32_t nCandidate, scaleImgStride, maskStride, sumStride;
    float sFactor;
    uint8_t *scaleImg = NULL, *mask = NULL, *srcHisteq = NULL;
    const uint16_t *ptr;
    uint16_t *sum; 
    void* buf = buffer;
    icvRect *candidateResult = NULL;
    ifcvFeature *feature = NULL;
    int32_t resultRunFearure, processingRectSizeWidth, processingRectSizeHeight; 
    uint32_t windowSizeWidth, windowSizeHeight, scaleImgWidth, scaleImgHeight;
    uint32_t foundCandidate;
    int32_t confidance;
    icvRect cf;
    uint32_t minNeighbors = FACE_MIN_NEIGHBOR;

    minFaceSize = CV_MAX(FACE_MODEL_X_SIZE, para->minSize);
    if( para->maxSize == 0 ){
        maxFaceSizeWidth = srcWidth;
        maxFaceSizeHeight = srcHeight;
    }
    else{
        maxFaceSizeWidth = para->maxSize;
        maxFaceSizeHeight = para->maxSize;
    }

    stepSize = para->stepSize;
    candidateResult = (icvRect*)buf;
    buf = (uint8_t*)buf + maxDetectedFaceNum * FACE_MAX_CANDIDATE_RECT * sizeof( icvRect);

    srcHisteq = (uint8_t*)buf;
    buf = (uint8_t*)buf + srcStride * srcHeight * sizeof(uint8_t) + 16; //for 128 bit alignemnt
    srcHisteq = (uint8_t*)((size_t)srcHisteq + (16 - (size_t)srcHisteq % 16));
	icvhistogramEqualizeImage( src, srcWidth, srcHeight, srcStride, srcHisteq);
    feature = (ifcvFeature*)buf;
    buf = (uint8_t*)buf + FACE_MODEL_N_FEATURE * sizeof(ifcvFeature);
    nCandidate = 0;


    cout << "para->scaleFactor = " << para->scaleFactor << endl;
    cout << "para max size = " << maxFaceSizeWidth << endl;
    cout << "para min size = " << minFaceSize << endl; 


    for(sFactor = 1.f; ; sFactor *= para->scaleFactor )
    {
        windowSizeWidth = uint32_t((FACE_MODEL_X_SIZE * sFactor) + 0.5);
        windowSizeHeight = uint32_t((FACE_MODEL_Y_SIZE * sFactor) + 0.5);
        scaleImgWidth = uint32_t((srcWidth * (1/sFactor)) + 0.5);
        scaleImgHeight = uint32_t((srcHeight * (1/sFactor)) + 0.5);
        processingRectSizeWidth = scaleImgWidth - FACE_MODEL_X_SIZE +1 ;
        processingRectSizeHeight = scaleImgHeight - FACE_MODEL_Y_SIZE +1;
    
		if( processingRectSizeWidth <= 0 || processingRectSizeHeight <= 0 )
            break;
		
        if( windowSizeWidth > maxFaceSizeWidth || windowSizeHeight > maxFaceSizeHeight )
            break;
		
        if( windowSizeWidth < minFaceSize || windowSizeHeight < minFaceSize )
            continue;
		
        cout << "sFactor =  " << sFactor << endl;
	
        scaleImg = (uint8_t*)buf;
        scaleImgStride = scaleImgWidth;
      
        buf = (uint8_t*)buf + scaleImgStride * scaleImgHeight * sizeof(uint8_t) + 16; //128 bit aligned
        scaleImg = (uint8_t*)((size_t)scaleImg + (16 - (size_t)scaleImg % 16));


		if( sFactor !=1.f) 
		{
            cvScaleDownMNu8Q( srcHisteq, srcWidth, srcHeight, srcStride, scaleImg, scaleImgWidth, scaleImgHeight, scaleImgStride);
		}
        else{
            scaleImgStride = srcStride;
            memcpy(scaleImg, srcHisteq, (scaleImgStride * scaleImgHeight ));
        }

        sumStride= (scaleImgWidth+1) * sizeof(uint16_t);
        //sumStride = (sumStride + 16 - 1) & -16; 
        sum = (uint16_t*)buf;
        buf = (uint8_t*)buf + sumStride * (scaleImgHeight+1) + 16;
        sum = (uint16_t*)((size_t)sum + (16 - (size_t)sum % 16));
		
        cvIntegrateImageu8_v2C_16Bit( scaleImg, scaleImgWidth, scaleImgHeight, scaleImgStride, sum, sumStride);

        sumStride =  sumStride/sizeof(uint16_t);

        for( fi = 0; fi < FACE_MODEL_N_FEATURE; fi++ )
        {
            ptr = (const uint16_t*)sum;
            fX = featureRectX[fi];
            fY = featureRectY[fi]; 
            fH = featureRectH[fi];
            feature[fi].p[0] = ptr+ fX+(sumStride*fY);
            feature[fi].p[1] = feature[fi].p[0]+ sumStride*fH;
        }
        stepSize = sFactor > 2. ? 1 : stepSize;
 #ifdef MULTITHREAD
		int stripCount, stripSize;

        const int PTS_PER_THREAD = 3500;
        stripCount = ((processingRectSizeWidth/stepSize)*(processingRectSizeHeight + stepSize-1)/stepSize + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSizeHeight + stripCount - 1)/stripCount + stepSize-1)/stepSize)*stepSize;
		
		vector<pthread_t> threads(stripCount);
		vector<Range> r(stripCount);

        cout << "number of strip count =" << stripCount << endl;
		for(int t=0; t<stripCount; t++)
		{
		
			if(t < (stripCount - 1))
			{
				r[t].start = t * stripSize;
				r[t].end = r[t].start + stripSize - 1;
			} else{
				r[t].start = t * stripSize;
				r[t].end = (uint32_t)processingRectSizeHeight - 1;
			}

			r[t].sFactor = sFactor;
			r[t].processingRectSizeWidth = processingRectSizeWidth;
			r[t].sumStride = sumStride;
			r[t].stepSize = stepSize;
			r[t].windowSizeWidth = windowSizeWidth;
			r[t].windowSizeHeight = windowSizeHeight;
			r[t].feature = feature;

            cout << " now creating thread " << t << endl;
			int rc = pthread_create(&threads[t], NULL, processStrip, (void *)&r[t]);
            cout << " done creating thread " << t << endl;

			if (rc)
			{
				printf("ERROR; return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
		}
        void *status;

          for(int t=0; t<stripCount; t++) 
          {
                cout << "pthread exit " << t << endl;
                int rc = pthread_join(threads[t], NULL);
                cout << "done pthread exit " << t << endl;
                if (rc) 
                {
                    printf("ERROR; return code from pthread_join() is %d\n", rc);
                    exit(-1);
                }
          }
 #else
        //foundCandidate = 0;
        for(y = 0; y < (uint32_t)processingRectSizeHeight; y += stepSize)
        {
            for(x = 0; x < (uint32_t)processingRectSizeWidth; x += stepSize )
            {
       
                offset = y * (sumStride) + x;
                resultRunFearure = icvRunFeature( feature, offset, &confidance, sumStride,x);	
                if( resultRunFearure > 0 ){
                    cf.topx = uint32_t((x*sFactor) + 0.5);
                    cf.topy = uint32_t((y*sFactor) + 0.5);
                    cf.width = windowSizeWidth;
                    cf.height = windowSizeHeight;
                    candidateResult[nCandidate++] = cf;
                    //foundCandidate = 1;
                } else if(resultRunFearure==0) 
                	x += stepSize;
      		}
        }
#endif
        buf = (uint8_t*)buf - sumStride * (scaleImgHeight+1) * sizeof(uint16_t)  - 16;
        buf = (uint8_t*)buf - scaleImgWidth * scaleImgHeight * sizeof(uint8_t) - 16;
    }

    feature = NULL;	
    icvRect *buff=(icvRect *)buffer;
    uint32_t detFaceNum=icvGrouprectangular( candidateResult, nCandidate,minNeighbors, buff);
    if(detFaceNum<= maxDetectedFaceNum){
        *resultFaceNum=detFaceNum;
        for(uint32_t i=0;i<detFaceNum;i++)
        {
            result[i].topx=buff[i].topx;
            result[i].topy=buff[i].topy;
            result[i].height=buff[i].height;
            result[i].width=buff[i].width;
        }
    }
    else{
        icvPickFace(detFaceNum,maxDetectedFaceNum, buff,result);
        *resultFaceNum = maxDetectedFaceNum;
    }
    return 1;   
}


void icvhistogramEqualizeImage( const uint8_t * __restrict src, uint32_t srcWidth, uint32_t srcHeight, uint32_t srcStride, uint8_t * __restrict dst)
{
    uint32_t i, j;
    uint32_t histogram[256];
    const uint8_t *ptr;
    uint8_t *ptr_;
    float fConversion;

    memset(histogram, 0, (256 * sizeof(uint32_t)));

    for( i=0 ; i < srcHeight; i++){
        ptr = src + i*srcStride;
        for( j=0 ; j < srcWidth; j++)
            histogram[int32_t(ptr[j])] += 1;
    }

    for( i=1 ; i<256 ; i++)
        histogram[i] += histogram[i-1];

    fConversion = 255.0f/(1.0f*histogram[255]);
    for( i=0 ; i < srcHeight; i++)
    {
        ptr = src + i*srcStride;
        ptr_ = dst + i*srcStride;
        for( j=0 ; j < srcWidth; j++)
            ptr_[j] = uint8_t(((fConversion * histogram[uint32_t(ptr[j])])) + 0.5);
    }	
}

void cvIntegrateImageu8_v2C_16Bit( const uint8_t* __restrict src,
                                  unsigned int srcWidth, 
                                  unsigned int srcHeight,
                                  unsigned int srcStride,
                                  uint16_t* __restrict dst,
                                  unsigned int dstStride )
{
    unsigned int i, j;
    unsigned int sum;
    uint16_t* iimg;
    uint16_t* iimgPrev;
    const uint8_t* img;

    dstStride = (dstStride / sizeof(uint16_t));     //Convert from #bytes to #values

   
    img      = src;
    iimgPrev = dst;
    iimg     = dst + dstStride;
    *iimg++ = *iimgPrev++ = sum = 0;  // zero first column

    for ( j=0; j<srcWidth; ++j )
    {
        iimgPrev[j] = 0;
        sum    += img[j];
        iimg[j] = (uint16_t)sum;
    }

    // remaining rows
    iimgPrev = iimg;
    img  += srcStride;
    iimg += dstStride;

    for ( i=1; i<srcHeight; ++i )
    {
        iimg[-1] = sum = 0;  // zero first column

        for ( j=0; j<srcWidth; ++j )
        {
            sum    += img[j];
            iimg[j] = (uint16_t)(sum + iimgPrev[j]);
        }
        iimgPrev = iimg;
        img     += srcStride;
        iimg    += dstStride;
    }
}

#define MN_DIV_Q_BITS            32
static const uint32_t mn_div_table[401] =
{
    4294967295,4294967295,2147483648,1431655766,1073741824,858993460,715827883,613566757,536870912,477218589,429496730,390451573,357913942,330382100,
    306783379,286331154,268435456,252645136,238609295,226050911,214748365,204522253,195225787,186737709,178956971,171798692,165191050,159072863,
    153391690,148102321,143165577,138547333,134217728,130150525,126322568,122713352,119304648,116080198,113025456,110127367,107374183,104755300,
    102261127,99882961,97612894,95443718,93368855,91382283,89478486,87652394,85899346,84215046,82595525,81037119,79536432,78090315,76695845,75350304,
    74051161,72796056,71582789,70409300,69273667,68174085,67108864,66076420,65075263,64103990,63161284,62245903,61356676,60492498,59652324,58835169,
    58040099,57266231,56512728,55778797,55063684,54366675,53687092,53024288,52377650,51746594,51130564,50529028,49941481,49367441,48806447,48258060,
    47721859,47197443,46684428,46182445,45691142,45210183,44739243,44278014,43826197,43383509,42949673,42524429,42107523,41698712,41297763,40904451,
    40518560,40139882,39768216,39403370,39045158,38693400,38347923,38008561,37675152,37347542,37025581,36709123,36398028,36092163,35791395,35495598,
    35204650,34918434,34636834,34359739,34087043,33818641,33554432,33294321,33038210,32786010,32537632,32292988,32051995,31814573,31580642,31350127,
    31122952,30899046,30678338,30460761,30246249,30034737,29826162,29620465,29417585,29217465,29020050,28825284,28633116,28443493,28256364,28071682,
    27889399,27709467,27531842,27356480,27183338,27012373,26843546,26676816,26512144,26349493,26188825,26030105,25873297,25718368,25565282,25414008,
    25264514,25116768,24970741,24826401,24683721,24542671,24403224,24265352,24129030,23994231,23860930,23729102,23598722,23469767,23342214,23216040,
    23091223,22967740,22845571,22724695,22605092,22486740,22369622,22253717,22139007,22025474,21913099,21801865,21691755,21582751,21474837,21367997,
    21262215,21157475,21053762,20951060,20849356,20748635,20648882,20550083,20452226,20355296,20259280,20164166,20069941,19976593,19884108,19792477,
    19701685,19611723,19522579,19434242,19346700,19259944,19173962,19088744,19004281,18920561,18837576,18755316,18673771,18592933,18512791,18433337,
    18354562,18276457,18199014,18122225,18046082,17970575,17895698,17821442,17747799,17674763,17602325,17530479,17459217,17388532,17318417,17248865,
    17179870,17111424,17043522,16976156,16909321,16843010,16777216,16711936,16647161,16582886,16519105,16455814,16393005,16330675,16268816,16207424,
    16146494,16086020,16025998,15966422,15907287,15848588,15790321,15732481,15675064,15618063,15561476,15505298,15449523,15394149,15339169,15284582,
    15230381,15176563,15123125,15070061,15017369,14965043,14913081,14861479,14810233,14759338,14708793,14658592,14608733,14559212,14510025,14461170,
    14412642,14364440,14316558,14268995,14221747,14174810,14128182,14081860,14035841,13990122,13944700,13899571,13854734,13810185,13765921,13721941,
    13678240,13634817,13591669,13548793,13506187,13463848,13421773,13379961,13338408,13297113,13256072,13215284,13174747,13134457,13094413,13054612,
    13015053,12975733,12936649,12897800,12859184,12820798,12782641,12744711,12707004,12669521,12632257,12595213,12558384,12521771,12485371,12449181,
    12413201,12377428,12341861,12306497,12271336,12236375,12201612,12167047,12132676,12098500,12064515,12030721,11997116,11963698,11930465,11897417,
    11864551,11831866,11799361,11767034,11734884,11702909,11671107,11639478,11608020,11576732,11545612,11514658,11483870,11453247,11422786,11392487,
    11362348,11332368,11302546,11272881,11243370,11214014,11184811,11155760,11126859,11098107,11069504,11041048,11012737,10984572,10956550,10928670,
    10900933,10873335,10845878,10818558,10791376,10764330,10737419
};


void cvScaleDownMNu8Q(const uint8_t* __restrict src,
                      uint32_t                  srcWidth,
                      uint32_t                  srcHeight,
                      uint32_t                  srcStride,
                      uint8_t* __restrict       dst,
                      uint32_t                  dstWidth,
                      uint32_t                  dstHeight,
                      uint32_t                  dstStride)
{
	
    if (!(src && dst && srcWidth && srcHeight && dstWidth && dstHeight &&
        (srcStride==0 || srcStride >=srcWidth) && (dstStride == 0 || dstStride >= dstWidth)))
    {
        return; 
    }

    srcStride   = (srcStride  == 0)? (srcWidth)  : srcStride;
    dstStride   = (dstStride  == 0)? (dstWidth)  : dstStride;

    if (dstWidth * 20 < srcWidth || dstHeight * 20 < srcHeight)
    {
        return; 
    }

    const int raw_buf_size = 2 * srcWidth * sizeof(uint16_t) + dstWidth + (4 * 32);
    uint8_t *raw_buf = (uint8_t *)malloc(raw_buf_size);
    if (!raw_buf)
    {
        return; 
    }


    uint8_t *horiz_input_pixels = (uint8_t*)(((uint32_t)raw_buf + 31) & ~31);
    uint16_t *raster = (uint16_t*)(horiz_input_pixels + ((dstWidth + 31) & ~31));
    uint16_t *raster2 = (uint16_t*)(((uint32_t)(&raster[srcWidth]) + 31) & ~31);

    uint32_t count = 0;
    uint8_t step  = 0;
    uint8_t* p_step = horiz_input_pixels;

    // pre-compute how many src columns are averaged for each dst column.
    for (uint32_t i = 0; i < srcWidth; i++)
    {
        count += dstWidth;        // M
        step++;

        if (count >= srcWidth)     // N
        {
            *p_step++ = step;
            count -= srcWidth;
            step = 0;
        }
    }

    count = 0;
    step  = 0;
    uint32_t accum = 0;

    // compute how many src rows are averaged for each dst row. During loop, perform averaging and write output pixels.
    for (uint32_t i = 0; i < dstHeight; i++)
    {
        // for new dst rows, reset raster values to next src row.
        for (uint32_t j = 0; j < srcWidth; j++)
        {
            raster[j] = src[j];
        }
        count += dstHeight;        // M
        step++;
        src += srcStride;
        while (count < srcHeight)
        {
            // add next src row to raster values.
            for (uint32_t j = 0; j < srcWidth; j++)
            {
                raster[j] += src[j];
            }
            count += dstHeight;        // M
            step++;
            src += srcStride;
        }

        // accumulate columns out output a row of pixels.
        uint16_t *p_raster = raster;
        for (uint32_t j = 0; j < dstWidth; j++)
        {
            // accumulate the appropriate number of column sums, divide by number of source pixels used, and output pixel
            accum = *p_raster++;
            for (uint32_t k = 1; k < horiz_input_pixels[j]; k++)
            {
                accum += *p_raster++;
            }
            // source pixels contributing to accum are step rows * horiz_input_pixels[j] columns.
            dst[j] = (uint8_t)(((uint64_t)accum * mn_div_table[step * horiz_input_pixels[j]]) >> MN_DIV_Q_BITS);
        }
        count -= srcHeight;
        step = 0;
        dst += dstStride;
    }
    free(raw_buf);
}







