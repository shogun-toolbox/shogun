#include <gtest/gtest.h>

#ifdef HAVE_OPENCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
 
using namespace cv;
 
TEST(OpenCV, OpenCV_Integration_Test)
{
	Mat cvMat = Mat::eye(3,3,CV_64FC1);
	EXPECT_EQ (cvMat.at<double>(0,0),1);
	EXPECT_EQ (cvMat.at<double>(1,0),0);
}	
#endif // HAVE_OPENCV

