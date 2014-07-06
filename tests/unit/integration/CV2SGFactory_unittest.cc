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
#include <shogun/lib/OpenCV/CV2SGFactory.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

using namespace cv;
using namespace shogun;

template<typename CV_T, typename SG_T>void test_helper_matrix(CV2SGOptions option)
{
	index_t data[]= {1, 3, 4, 0, 1, 0, 0, 0, 1};
	const int cv_type=OpenCVTypeName<CV_T>::get_opencv_type();
	Mat cvMat=Mat::eye(3,3,cv_type);
	cvMat.at<CV_T>(0,1)=3;
	cvMat.at<CV_T>(0,2)=4;

	SGMatrix<SG_T> sgMat=CV2SGFactory::getSGMatrix<SG_T>(cvMat, option);
	index_t k=0;
	for (index_t i=0; i< cvMat.rows; ++i)
		for (index_t j=0; j< cvMat.cols; ++j)
		{
			EXPECT_EQ (sgMat(i,j),data[k]);
			++k;
		}
}

template<typename CV_T, typename SG_T>void test_helper_features(CV2SGOptions option)
{
	index_t data[]= {1, 3, 4, 0, 1, 0, 0, 0, 1};
	const int cv_type=OpenCVTypeName<CV_T>::get_opencv_type();
	Mat cvMat=Mat::eye(3,3,cv_type);
	cvMat.at<CV_T>(0,1)=3;
	cvMat.at<CV_T>(0,2)=4;

	CDenseFeatures<SG_T>* sgDense=CV2SGFactory::getDensefeatures<SG_T>(cvMat, option);
	SGMatrix<SG_T> sgMat=sgDense->get_feature_matrix(); 
	index_t k=0;
	for (index_t i=0; i< cvMat.rows; ++i)
		for (index_t j=0; j< cvMat.cols; ++j)
		{
			EXPECT_EQ (sgMat(i,j),data[k]);
			++k;
		}
}

//***************************************************************************//
//						  CVMat2SGMatrix unittest							 //
//***************************************************************************// 

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_uint8)
{
	test_helper_matrix<unsigned char, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, uint8_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, uint8_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_int8)
{
	test_helper_matrix<unsigned char, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, int8_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, int8_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_uint16)
{
	test_helper_matrix<unsigned char, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, uint16_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, uint16_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_int16)
{
	test_helper_matrix<unsigned char, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, int16_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, int16_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_int32)
{
	test_helper_matrix<unsigned char, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, int32_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, int32_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_float32)
{
	test_helper_matrix<unsigned char, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, float32_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, float32_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MEMCPY, test_float64)
{
	test_helper_matrix<unsigned char, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed char, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<unsigned short, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<signed short, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<int, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<float, float64_t>(CV2SG_MEMCPY);
	test_helper_matrix<double, float64_t>(CV2SG_MEMCPY);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_uint8)
{
	test_helper_matrix<unsigned char, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<int, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<float, uint8_t>(CV2SG_MANUAL);
	test_helper_matrix<double, uint8_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_int8)
{
	test_helper_matrix<unsigned char, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<int, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<float, int8_t>(CV2SG_MANUAL);
	test_helper_matrix<double, int8_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_uint16)
{
	test_helper_matrix<unsigned char, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<int, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<float, uint16_t>(CV2SG_MANUAL);
	test_helper_matrix<double, uint16_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_int16)
{
	test_helper_matrix<unsigned char, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<int, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<float, int16_t>(CV2SG_MANUAL);
	test_helper_matrix<double, int16_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_int32)
{
	test_helper_matrix<unsigned char, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<int, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<float, int32_t>(CV2SG_MANUAL);
	test_helper_matrix<double, int32_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_float32)
{
	test_helper_matrix<unsigned char, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<int, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<float, float32_t>(CV2SG_MANUAL);
	test_helper_matrix<double, float32_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_MANUAL, test_float64)
{
	test_helper_matrix<unsigned char, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<signed char, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<unsigned short, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<signed short, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<int, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<float, float64_t>(CV2SG_MANUAL);
	test_helper_matrix<double, float64_t>(CV2SG_MANUAL);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_uint8)
{
	test_helper_matrix<unsigned char, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, uint8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, uint8_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_int8)
{
	test_helper_matrix<unsigned char, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, int8_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, int8_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_uint16)
{
	test_helper_matrix<unsigned char, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, uint16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, uint16_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_int16)
{
	test_helper_matrix<unsigned char, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, int16_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, int16_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_int32)
{
	test_helper_matrix<unsigned char, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, int32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, int32_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_float32)
{
	test_helper_matrix<unsigned char, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, float32_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, float32_t>(CV2SG_CONSTRUCTOR);
}

TEST(CVMat2SGMatrix_CV2SG_CONSTRUCTOR, test_float64)
{
	test_helper_matrix<unsigned char, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed char, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<unsigned short, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<signed short, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<int, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<float, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_matrix<double, float64_t>(CV2SG_CONSTRUCTOR);
}

//***************************************************************************//
//						  CVMat2DenseFeature								 //
//***************************************************************************//

TEST(CVMat2DenseFeatures_CV2SG_MEMCPY, test_uint16)
{
	test_helper_features<unsigned char, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<signed char, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<unsigned short, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<signed short, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<int, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<float, uint16_t>(CV2SG_MEMCPY);
	test_helper_features<double, uint16_t>(CV2SG_MEMCPY);
}

TEST(CVMat2DenseFeatures_CV2SG_MANUAL, test_int32)
{
	test_helper_features<unsigned char, int32_t>(CV2SG_MANUAL);
	test_helper_features<signed char, int32_t>(CV2SG_MANUAL);
	test_helper_features<unsigned short, int32_t>(CV2SG_MANUAL);
	test_helper_features<signed short, int32_t>(CV2SG_MANUAL);
	test_helper_features<int, int32_t>(CV2SG_MANUAL);
	test_helper_features<float, int32_t>(CV2SG_MANUAL);
	test_helper_features<double, int32_t>(CV2SG_MANUAL);
}

TEST(CVMat2DenseFeatures_CV2SG_CONSTRUCTOR, test_float64)
{
	test_helper_features<unsigned char, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<signed char, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<unsigned short, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<signed short, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<int, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<float, float64_t>(CV2SG_CONSTRUCTOR);
	test_helper_features<double, float64_t>(CV2SG_CONSTRUCTOR);
}
#endif //HAVE_OPENCV
