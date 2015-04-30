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
#include <shogun/lib/OpenCV/SG2CVFactory.h>

using namespace cv;
using namespace shogun;

template<typename SG_T>void test_helper(int cv_type)
{
	index_t expected[]={1, 3, 4, 0, 1, 0, 0, 0, 1};

	SGMatrix<SG_T> sgMat=SGMatrix<SG_T>::create_identity_matrix(3, 1);
	sgMat(0,1)=3;
	sgMat(0,2)=4;
	CDenseFeatures<SG_T>* sgDense=new CDenseFeatures<SG_T>(sgMat);
	SG_REF(sgDense)
	Mat cvMat1=SG2CVFactory::get_cvMat_from_features<SG_T>(sgDense, cv_type);
	Mat cvMat2=SG2CVFactory::get_cvMat<SG_T>(sgMat, cv_type);
	index_t k=0;
	for (index_t i=0; i<3; ++i)
	{
		for (index_t j=0; j<3; ++j)
		{
			switch(cv_type)
			{
				case CV_8U:
					EXPECT_EQ(cvMat1.at<unsigned char>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<unsigned char>(i,j), expected[k]);
					break;

				case CV_8S:
					EXPECT_EQ(cvMat1.at<signed char>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<signed char>(i,j), expected[k]);
					break;

				case CV_16U:
					EXPECT_EQ(cvMat1.at<unsigned short>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<unsigned short>(i,j), expected[k]);
					break;

				case CV_16S:
					EXPECT_EQ(cvMat1.at<signed short>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<signed short>(i,j), expected[k]);
					break;

				case CV_32S:
					EXPECT_EQ(cvMat1.at<int>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<int>(i,j), expected[k]);
					break;

				case CV_32F:
					EXPECT_EQ(cvMat1.at<float>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<float>(i,j), expected[k]);
					break;

				case CV_64F:
					EXPECT_EQ(cvMat1.at<double>(i,j), expected[k]);
					EXPECT_EQ(cvMat2.at<double>(i,j), expected[k]);
					break;
			}
			++k;
		}
	}
	SG_UNREF(sgDense)
}

TEST(SG2CVFactory, test_CV_8U)
{
	test_helper<uint8_t>(CV_8U);
	test_helper<int8_t>(CV_8U);
	test_helper<uint16_t>(CV_8U);
	test_helper<int16_t>(CV_8U);
	test_helper<int32_t>(CV_8U);
	test_helper<float32_t>(CV_8U);
	test_helper<float64_t>(CV_8U);
}

TEST(SG2CVFactory, test_CV_8S)
{
	test_helper<uint8_t>(CV_8S);
	test_helper<int8_t>(CV_8S);
	test_helper<uint16_t>(CV_8S);
	test_helper<int16_t>(CV_8S);
	test_helper<int32_t>(CV_8S);
	test_helper<float32_t>(CV_8S);
	test_helper<float64_t>(CV_8S);
}

TEST(SG2CVFactory, test_CV_16U)
{
	test_helper<uint8_t>(CV_16U);
	test_helper<int8_t>(CV_16U);
	test_helper<uint16_t>(CV_16U);
	test_helper<int16_t>(CV_16U);
	test_helper<int32_t>(CV_16U);
	test_helper<float32_t>(CV_16U);
	test_helper<float64_t>(CV_16U);
}

TEST(SG2CVFactory, test_CV_16S)
{
	test_helper<uint8_t>(CV_16S);
	test_helper<int8_t>(CV_16S);
	test_helper<uint16_t>(CV_16S);
	test_helper<int16_t>(CV_16S);
	test_helper<int32_t>(CV_16S);
	test_helper<float32_t>(CV_16S);
	test_helper<float64_t>(CV_16S);
}

TEST(SG2CVFactory, test_CV_32S)
{
	test_helper<uint8_t>(CV_32S);
	test_helper<int8_t>(CV_32S);
	test_helper<uint16_t>(CV_32S);
	test_helper<int16_t>(CV_32S);
	test_helper<int32_t>(CV_32S);
	test_helper<float32_t>(CV_32S);
	test_helper<float64_t>(CV_32S);
}

TEST(SG2CVFactory, test_CV_32F)
{
	test_helper<uint8_t>(CV_32F);
	test_helper<int8_t>(CV_32F);
	test_helper<uint16_t>(CV_32F);
	test_helper<int16_t>(CV_32F);
	test_helper<int32_t>(CV_32F);
	test_helper<float32_t>(CV_32F);
	test_helper<float64_t>(CV_32F);
}

TEST(SG2CVFactory, test_CV_64F)
{
	test_helper<uint8_t>(CV_64F);
	test_helper<int8_t>(CV_64F);
	test_helper<uint16_t>(CV_64F);
	test_helper<int16_t>(CV_64F);
	test_helper<int32_t>(CV_64F);
	test_helper<float32_t>(CV_64F);
	test_helper<float64_t>(CV_64F);
}
#endif //HAVE_OPENCV
