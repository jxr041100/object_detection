echo off
set ROOT=..\..\
set OPENCV_ROOT=C:\Ashwath\openCV\bin\Release 


echo ****create the positive vectors based on positive samples*****
C:\Ashwath\openCV\bin\Release\opencv_createsamples -info C:\Ashwath\data\face\Fulldata_15-15.dat -w 20 -h 20 -num 175609 -vec C:\Ashwath\data\face\vecOpenCVFace15x15.vec




