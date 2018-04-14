/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Abhijeet Kislay, Bjoern Esser
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
