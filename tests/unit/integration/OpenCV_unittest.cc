/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Abhijeet Kislay
 */

#include <shogun/lib/config.h>
#ifdef HAVE_OPENCV

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
 
using namespace cv;

TEST(OpenCV, OpenCV_Integration_Test)
{
	Mat cvMat = Mat::eye(3,3,CV_64FC1);
	EXPECT_EQ (cvMat.at<double>(0,0),1);
	EXPECT_EQ (cvMat.at<double>(1,0),0);
}	
#endif // HAVE_OPENCV
