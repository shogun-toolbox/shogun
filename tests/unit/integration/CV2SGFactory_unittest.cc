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
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/OpenCV/CV2SGFactory.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

using namespace cv;
using namespace shogun;

template<typename CV_T, typename SG_T>void test_helper_matrix()
{
	index_t data[]= {1, 3, 4, 0, 1, 0, 0, 0, 1};
	const int cv_type=OpenCVTypeName<CV_T>::get_opencv_type();
	Mat cvMat=Mat::eye(3,3,cv_type);
	cvMat.at<CV_T>(0,1)=3;
	cvMat.at<CV_T>(0,2)=4;

	SGMatrix<SG_T> sgMat1=CV2SGFactory::get_sgmatrix<SG_T>(cvMat);
	index_t k=0;
	for (index_t i=0; i< cvMat.rows; ++i)
		for (index_t j=0; j< cvMat.cols; ++j)
		{
			EXPECT_EQ (sgMat1(i,j),data[k]);
			++k;
		}

	CDenseFeatures<SG_T>* sgDense=CV2SGFactory::get_dense_features<SG_T>(cvMat);
	SG_REF(sgDense)
	SGMatrix<SG_T> sgMat2=sgDense->get_feature_matrix();
	k=0;
	for (index_t i=0; i<cvMat.rows; ++i)
		for (index_t j=0; j<cvMat.cols; ++j)
		{
			EXPECT_EQ (sgMat2(i,j),data[k]);
			++k;
		}
	SG_UNREF(sgDense)
}

TEST(CV2SGFactory, test_uint8)
{
	test_helper_matrix<unsigned char, uint8_t>();
	test_helper_matrix<signed char, uint8_t>();
	test_helper_matrix<unsigned short, uint8_t>();
	test_helper_matrix<signed short, uint8_t>();
	test_helper_matrix<int, uint8_t>();
	test_helper_matrix<float, uint8_t>();
	test_helper_matrix<double, uint8_t>();
}

TEST(CV2SGFactory, test_int8)
{
	test_helper_matrix<unsigned char, int8_t>();
	test_helper_matrix<signed char, int8_t>();
	test_helper_matrix<unsigned short, int8_t>();
	test_helper_matrix<signed short, int8_t>();
	test_helper_matrix<int, int8_t>();
	test_helper_matrix<float, int8_t>();
	test_helper_matrix<double, int8_t>();
}

TEST(CV2SGFactory, test_uint16)
{
	test_helper_matrix<unsigned char, uint16_t>();
	test_helper_matrix<signed char, uint16_t>();
	test_helper_matrix<unsigned short, uint16_t>();
	test_helper_matrix<signed short, uint16_t>();
	test_helper_matrix<int, uint16_t>();
	test_helper_matrix<float, uint16_t>();
	test_helper_matrix<double, uint16_t>();
}

TEST(CV2SGFactory, test_int16)
{
	test_helper_matrix<unsigned char, int16_t>();
	test_helper_matrix<signed char, int16_t>();
	test_helper_matrix<unsigned short, int16_t>();
	test_helper_matrix<signed short, int16_t>();
	test_helper_matrix<int, int16_t>();
	test_helper_matrix<float, int16_t>();
	test_helper_matrix<double, int16_t>();
}

TEST(CV2SGFactory, test_int32)
{
	test_helper_matrix<unsigned char, int32_t>();
	test_helper_matrix<signed char, int32_t>();
	test_helper_matrix<unsigned short, int32_t>();
	test_helper_matrix<signed short, int32_t>();
	test_helper_matrix<int, int32_t>();
	test_helper_matrix<float, int32_t>();
	test_helper_matrix<double, int32_t>();
}

TEST(CV2SGFactory, test_float32)
{
	test_helper_matrix<unsigned char, float32_t>();
	test_helper_matrix<signed char, float32_t>();
	test_helper_matrix<unsigned short, float32_t>();
	test_helper_matrix<signed short, float32_t>();
	test_helper_matrix<int, float32_t>();
	test_helper_matrix<float, float32_t>();
	test_helper_matrix<double, float32_t>();
}

TEST(CV2SGFactory, test_float64)
{
	test_helper_matrix<unsigned char, float64_t>();
	test_helper_matrix<signed char, float64_t>();
	test_helper_matrix<unsigned short, float64_t>();
	test_helper_matrix<signed short, float64_t>();
	test_helper_matrix<int, float64_t>();
	test_helper_matrix<float, float64_t>();
	test_helper_matrix<double, float64_t>();
}
#endif //HAVE_OPENCV
