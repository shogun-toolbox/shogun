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
#include <opencv2/highgui/highgui.hpp>

#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/CV2FeaturesFactory.h>
#include <shogun/lib/OpenCV/SG2CVMatFactory.h>


using namespace cv;
using namespace shogun;

class Test_Helper_Class
{   
    public:        
    template <typename T> static void test_Helper_Function(Mat,SG2CVOptions);
};

template<typename T>void Test_Helper_Class::test_Helper_Function(Mat cvMat, SG2CVOptions option)
{      
	index_t data[] =  {1, 3, 4, 0, 1, 0, 0, 0, 1};
   	CDenseFeatures<T>* A = CV2FeaturesFactory::getDenseFeatures<T>(cvMat,CV2SG_MANUAL);
     
    Mat cvMat1 = SG2CVMatFactory::getcvMat<T,uint8_t>(A, option);
    Mat cvMat2 = SG2CVMatFactory::getcvMat<T,int8_t>(A, option);
    Mat cvMat3 = SG2CVMatFactory::getcvMat<T,uint16_t>(A, option);
    Mat cvMat4 = SG2CVMatFactory::getcvMat<T,int16_t>(A, option);
    Mat cvMat5 = SG2CVMatFactory::getcvMat<T,int32_t>(A, option);
    Mat cvMat6 = SG2CVMatFactory::getcvMat<T,float32_t>(A, option);
    Mat cvMat7 = SG2CVMatFactory::getcvMat<T,float64_t>(A, option);

	index_t k=0;
	for (index_t i=0; i<3; ++i)
	{
		for (index_t j=0; j<3; ++j)
		{
			EXPECT_EQ(cvMat1.at<uint8_t>(i,j), data[k]);
			EXPECT_EQ(cvMat2.at<int8_t>(i,j), data[k]);
			EXPECT_EQ(cvMat3.at<uint16_t>(i,j), data[k]);
			EXPECT_EQ(cvMat4.at<int16_t>(i,j), data[k]);
			EXPECT_EQ(cvMat5.at<int32_t>(i,j), data[k]);
			EXPECT_EQ(cvMat6.at<float32_t>(i,j), data[k]);
			EXPECT_EQ(cvMat7.at<float64_t>(i,j), data[k]);
			++k;
		}
	}
}

TEST(SGMatrix2CVMatManual, SGMatrix_to_CVMat_conversion_using_Manual)
{
    Mat cvMat = Mat::eye(3,3,CV_8U);
	cvMat.at<unsigned char>(0,1) =3;
    cvMat.at<unsigned char>(0,2) =4;

    Test_Helper_Class::test_Helper_Function<uint8_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<int8_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<uint16_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<int16_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<int32_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<float32_t>(cvMat, SG2CV_MANUAL);
    Test_Helper_Class::test_Helper_Function<float64_t>(cvMat, SG2CV_MANUAL);
    }

TEST(SGMatrix2CVMatMemcpy, SGMatrix_to_CVMat_conversion_using_Memcpy)
{
    Mat cvMat = Mat::eye(3,3,CV_8U);
	cvMat.at<unsigned char>(0,1) =3;
    cvMat.at<unsigned char>(0,2) =4;

    Test_Helper_Class::test_Helper_Function<uint8_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int8_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<uint16_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int16_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int32_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<float32_t>(cvMat, SG2CV_MEMCPY);
    Test_Helper_Class::test_Helper_Function<float64_t>(cvMat, SG2CV_MEMCPY);
}
#endif //HAVE_OPENCV
