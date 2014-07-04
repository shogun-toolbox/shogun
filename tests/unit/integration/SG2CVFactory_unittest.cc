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
#include <shogun/lib/OpenCV/SG2CVFactory.h>

using namespace cv;
using namespace shogun;

template<typename SG_T, typename CV_T>void test_helper_matrix(SG2CVOptions option)
{
	index_t expected[]={1, 3, 4, 0, 1, 0, 0, 0, 1};
	SGMatrix<SG_T> sgMat=SGMatrix<SG_T>::create_identity_matrix(3, 1);
	sgMat(0,1)=3;
	sgMat(0,2)=4;
	Mat cvMat=SG2CVFactory::getcvMat<SG_T,CV_T>(sgMat, option);
	index_t k=0;
	for (index_t i=0; i<3; ++i)
		for (index_t j=0; j<3; ++j)
		{
			EXPECT_EQ(cvMat.at< CV_T >(i,j), expected[k]);
			++k;
		}
}

template<typename SG_T>void test_helper_matrix(int cv_type, SG2CVOptions option)
{
	index_t expected[]={1, 3, 4, 0, 1, 0, 0, 0, 1};

	SGMatrix<SG_T> sgMat=SGMatrix<SG_T>::create_identity_matrix(3, 1);
	sgMat(0,1)=3;
	sgMat(0,2)=4;
	Mat cvMat=SG2CVFactory::getcvMat<SG_T>(sgMat, cv_type, option);

	index_t k=0;
	for (index_t i=0; i<3; ++i)
	{
		for (index_t j=0; j<3; ++j)
		{
			switch(cv_type)
			{
				case CV_8U:
					EXPECT_EQ(cvMat.at< unsigned char >(i,j), expected[k]);
					break;

				case CV_8S:
					EXPECT_EQ(cvMat.at< signed char >(i,j), expected[k]);
					break;

				case CV_16U:
					EXPECT_EQ(cvMat.at< unsigned short >(i,j), expected[k]);
					break;

				case CV_16S:
					EXPECT_EQ(cvMat.at< signed short >(i,j), expected[k]);
					break;

				case CV_32S:
					EXPECT_EQ(cvMat.at< int >(i,j), expected[k]);
					break;

				case CV_32F:
					EXPECT_EQ(cvMat.at< float >(i,j), expected[k]);
					break;

				case CV_64F:
					EXPECT_EQ(cvMat.at< double >(i,j), expected[k]);
					break;
			}
			++k;
		}
	}
}

template<typename SG_T, typename CV_T>void test_helper_features(SG2CVOptions option)
{
	index_t expected[]={1, 3, 4, 0, 1, 0, 0, 0, 1};
	SGMatrix<SG_T> sgMat=SGMatrix<SG_T>::create_identity_matrix(3, 1);
	sgMat(0,1)=3;
	sgMat(0,2)=4;
	CDenseFeatures<SG_T>* sgDense=new CDenseFeatures<SG_T>(sgMat);
	Mat cvMat=SG2CVFactory::getcvMat_from_features<SG_T,CV_T>(sgDense, option);
	index_t k=0;
	for (index_t i=0; i<3; ++i)
		for (index_t j=0; j<3; ++j)
		{
			EXPECT_EQ(cvMat.at< CV_T >(i,j), expected[k]);
			++k;
		}
}

template<typename SG_T>void test_helper_features(int cv_type, SG2CVOptions option)
{
	index_t expected[]={1, 3, 4, 0, 1, 0, 0, 0, 1};
	SGMatrix<SG_T> sgMat=SGMatrix<SG_T>::create_identity_matrix(3, 1);
	sgMat(0,1)=3;
	sgMat(0,2)=4;
	CDenseFeatures<SG_T>* sgDense=new CDenseFeatures<SG_T>(sgMat);
	Mat cvMat=SG2CVFactory::getcvMat_from_features<SG_T>(sgDense, cv_type, option);
	index_t k=0;
	for (index_t i=0; i<3; ++i)
	{
		for (index_t j=0; j<3; ++j)
		{
			switch(cv_type)
			{
				case CV_8U:
					EXPECT_EQ(cvMat.at< unsigned char >(i,j), expected[k]);
					break;

				case CV_8S:
					EXPECT_EQ(cvMat.at< signed char >(i,j), expected[k]);
					break;

				case CV_16U:
					EXPECT_EQ(cvMat.at< unsigned short >(i,j), expected[k]);
					break;

				case CV_16S:
					EXPECT_EQ(cvMat.at< signed short >(i,j), expected[k]);
					break;

				case CV_32S:
					EXPECT_EQ(cvMat.at< int >(i,j), expected[k]);
					break;

				case CV_32F:
					EXPECT_EQ(cvMat.at< float >(i,j), expected[k]);
					break;

				case CV_64F:
					EXPECT_EQ(cvMat.at< double >(i,j), expected[k]);
					break;
			}
			++k;
		}
	}
}

//**********************************************************//
//			 SGMatrix2CVMat unit-test						//
//**********************************************************// 
TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_8U)
{
	test_helper_matrix<uint8_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_8U, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_unsigned_char)
{
	test_helper_matrix<uint8_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, unsigned char>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_8S)
{
	test_helper_matrix<uint8_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_8S, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_signed_char)
{
	test_helper_matrix<uint8_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, signed char>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, signed char>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_16U)
{
	test_helper_matrix<uint8_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_16U, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_unsigned_short)
{
	test_helper_matrix<uint8_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, unsigned short>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_16S)
{
	test_helper_matrix<uint8_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_16S, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_signed_short)
{
	test_helper_matrix<uint8_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, signed short>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, signed short>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_32S)
{
	test_helper_matrix<uint8_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_32S, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_integer)
{
	test_helper_matrix<uint8_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, int>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, int>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_32F)
{
	test_helper_matrix<uint8_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_32F, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_float)
{
	test_helper_matrix<uint8_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, float>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, float>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_CV_64F)
{
	test_helper_matrix<uint8_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<int8_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<uint16_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<int16_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<int32_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<float32_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_matrix<float64_t>(CV_64F, SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MEMCPY, test_double)
{
	test_helper_matrix<uint8_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<int8_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<uint16_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<int16_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<int32_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<float32_t, double>(SG2CV_MEMCPY);
	test_helper_matrix<float64_t, double>(SG2CV_MEMCPY);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_8U)
{
	test_helper_matrix<uint8_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_8U, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_8U, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_unsigned_char)
{
	test_helper_matrix<uint8_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, unsigned char>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, unsigned char>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_8S)
{
	test_helper_matrix<uint8_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_8S, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_8S, SG2CV_MANUAL);
}


TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_signed_char)
{
	test_helper_matrix<uint8_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, signed char>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, signed char>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_16U)
{
	test_helper_matrix<uint8_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_16U, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_16U, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_unsigned_short)
{
	test_helper_matrix<uint8_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, unsigned short>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, unsigned short>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_16S)
{
	test_helper_matrix<uint8_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_16S, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_16S, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_signed_short)
{
	test_helper_matrix<uint8_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, signed short>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, signed short>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_32S)
{
	test_helper_matrix<uint8_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_32S, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_32S, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_integer)
{
	test_helper_matrix<uint8_t, int>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, int>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, int>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, int>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, int>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, int>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, int>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_32F)
{
	test_helper_matrix<uint8_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_32F, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_32F, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_float)
{
	test_helper_matrix<uint8_t, float>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, float>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, float>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, float>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, float>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, float>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, float>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_CV_64F)
{
	test_helper_matrix<uint8_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<int8_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<uint16_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<int16_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<int32_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<float32_t>(CV_64F, SG2CV_MANUAL);
	test_helper_matrix<float64_t>(CV_64F, SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_MANUAL, test_double)
{
	test_helper_matrix<uint8_t, double>(SG2CV_MANUAL);
	test_helper_matrix<int8_t, double>(SG2CV_MANUAL);
	test_helper_matrix<uint16_t, double>(SG2CV_MANUAL);
	test_helper_matrix<int16_t, double>(SG2CV_MANUAL);
	test_helper_matrix<int32_t, double>(SG2CV_MANUAL);
	test_helper_matrix<float32_t, double>(SG2CV_MANUAL);
	test_helper_matrix<float64_t, double>(SG2CV_MANUAL);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_8U)
{
	test_helper_matrix<uint8_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_8U, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_unsigned_char)
{
	test_helper_matrix<uint8_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, unsigned char>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_8S)
{
	test_helper_matrix<uint8_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_8S, SG2CV_CONSTRUCTOR);
}


TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_signed_char)
{
	test_helper_matrix<uint8_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, signed char>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_16U)
{
	test_helper_matrix<uint8_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_16U, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_unsigned_short)
{
	test_helper_matrix<uint8_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, unsigned short>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_16S)
{
	test_helper_matrix<uint8_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_16S, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_signed_short)
{
	test_helper_matrix<uint8_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, signed short>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_32S)
{
	test_helper_matrix<uint8_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_32S, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_integer)
{
	test_helper_matrix<uint8_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, int>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_32F)
{
	test_helper_matrix<uint8_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_32F, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_float)
{
	test_helper_matrix<uint8_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, float>(SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_64F)
{
	test_helper_matrix<uint8_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t>(CV_64F, SG2CV_CONSTRUCTOR);
}

TEST(SGMatrix2CVMat_using_SG2CV_CONSTRUCTOR, test_double)
{
	test_helper_matrix<uint8_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int8_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<uint16_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int16_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<int32_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float32_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_matrix<float64_t, double>(SG2CV_CONSTRUCTOR);
}

//**********************************************************//
//		DenseFeature2CVMat unit-test						//
//**********************************************************// 

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_8U)
{
	test_helper_features<uint8_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_8U, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_8U, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_unsigned_char)
{
	test_helper_features<uint8_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<int8_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<int16_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<int32_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<float32_t, unsigned char>(SG2CV_MEMCPY);
	test_helper_features<float64_t, unsigned char>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_8S)
{
	test_helper_features<uint8_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_8S, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_8S, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_signed_char)
{
	test_helper_features<uint8_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<int8_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<int16_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<int32_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<float32_t, signed char>(SG2CV_MEMCPY);
	test_helper_features<float64_t, signed char>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_16U)
{
	test_helper_features<uint8_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_16U, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_16U, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_unsigned_short)
{
	test_helper_features<uint8_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<int8_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<int16_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<int32_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<float32_t, unsigned short>(SG2CV_MEMCPY);
	test_helper_features<float64_t, unsigned short>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_16S)
{
	test_helper_features<uint8_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_16S, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_16S, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_signed_short)
{
	test_helper_features<uint8_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<int8_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<int16_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<int32_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<float32_t, signed short>(SG2CV_MEMCPY);
	test_helper_features<float64_t, signed short>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_32S)
{
	test_helper_features<uint8_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_32S, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_32S, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_integer)
{
	test_helper_features<uint8_t, int>(SG2CV_MEMCPY);
	test_helper_features<int8_t, int>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, int>(SG2CV_MEMCPY);
	test_helper_features<int16_t, int>(SG2CV_MEMCPY);
	test_helper_features<int32_t, int>(SG2CV_MEMCPY);
	test_helper_features<float32_t, int>(SG2CV_MEMCPY);
	test_helper_features<float64_t, int>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_32F)
{
	test_helper_features<uint8_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_32F, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_32F, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_float)
{
	test_helper_features<uint8_t, float>(SG2CV_MEMCPY);
	test_helper_features<int8_t, float>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, float>(SG2CV_MEMCPY);
	test_helper_features<int16_t, float>(SG2CV_MEMCPY);
	test_helper_features<int32_t, float>(SG2CV_MEMCPY);
	test_helper_features<float32_t, float>(SG2CV_MEMCPY);
	test_helper_features<float64_t, float>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_CV_64F)
{
	test_helper_features<uint8_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<int8_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<uint16_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<int16_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<int32_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<float32_t>(CV_64F, SG2CV_MEMCPY);
	test_helper_features<float64_t>(CV_64F, SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MEMCPY, test_double)
{
	test_helper_features<uint8_t, double>(SG2CV_MEMCPY);
	test_helper_features<int8_t, double>(SG2CV_MEMCPY);
	test_helper_features<uint16_t, double>(SG2CV_MEMCPY);
	test_helper_features<int16_t, double>(SG2CV_MEMCPY);
	test_helper_features<int32_t, double>(SG2CV_MEMCPY);
	test_helper_features<float32_t, double>(SG2CV_MEMCPY);
	test_helper_features<float64_t, double>(SG2CV_MEMCPY);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_8U)
{
	test_helper_features<uint8_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_8U, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_8U, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_unsigned_char)
{
	test_helper_features<uint8_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<int8_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<uint16_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<int16_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<int32_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<float32_t, unsigned char>(SG2CV_MANUAL);
	test_helper_features<float64_t, unsigned char>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_8S)
{
	test_helper_features<uint8_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_8S, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_8S, SG2CV_MANUAL);
}


TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_signed_char)
{
	test_helper_features<uint8_t, signed char>(SG2CV_MANUAL);
	test_helper_features<int8_t, signed char>(SG2CV_MANUAL);
	test_helper_features<uint16_t, signed char>(SG2CV_MANUAL);
	test_helper_features<int16_t, signed char>(SG2CV_MANUAL);
	test_helper_features<int32_t, signed char>(SG2CV_MANUAL);
	test_helper_features<float32_t, signed char>(SG2CV_MANUAL);
	test_helper_features<float64_t, signed char>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_16U)
{
	test_helper_features<uint8_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_16U, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_16U, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_unsigned_short)
{
	test_helper_features<uint8_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<int8_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<uint16_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<int16_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<int32_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<float32_t, unsigned short>(SG2CV_MANUAL);
	test_helper_features<float64_t, unsigned short>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_16S)
{
	test_helper_features<uint8_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_16S, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_16S, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_signed_short)
{
	test_helper_features<uint8_t, signed short>(SG2CV_MANUAL);
	test_helper_features<int8_t, signed short>(SG2CV_MANUAL);
	test_helper_features<uint16_t, signed short>(SG2CV_MANUAL);
	test_helper_features<int16_t, signed short>(SG2CV_MANUAL);
	test_helper_features<int32_t, signed short>(SG2CV_MANUAL);
	test_helper_features<float32_t, signed short>(SG2CV_MANUAL);
	test_helper_features<float64_t, signed short>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_32S)
{
	test_helper_features<uint8_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_32S, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_32S, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_integer)
{
	test_helper_features<uint8_t, int>(SG2CV_MANUAL);
	test_helper_features<int8_t, int>(SG2CV_MANUAL);
	test_helper_features<uint16_t, int>(SG2CV_MANUAL);
	test_helper_features<int16_t, int>(SG2CV_MANUAL);
	test_helper_features<int32_t, int>(SG2CV_MANUAL);
	test_helper_features<float32_t, int>(SG2CV_MANUAL);
	test_helper_features<float64_t, int>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_32F)
{
	test_helper_features<uint8_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_32F, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_32F, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_float)
{
	test_helper_features<uint8_t, float>(SG2CV_MANUAL);
	test_helper_features<int8_t, float>(SG2CV_MANUAL);
	test_helper_features<uint16_t, float>(SG2CV_MANUAL);
	test_helper_features<int16_t, float>(SG2CV_MANUAL);
	test_helper_features<int32_t, float>(SG2CV_MANUAL);
	test_helper_features<float32_t, float>(SG2CV_MANUAL);
	test_helper_features<float64_t, float>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_CV_64F)
{
	test_helper_features<uint8_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<int8_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<uint16_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<int16_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<int32_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<float32_t>(CV_64F, SG2CV_MANUAL);
	test_helper_features<float64_t>(CV_64F, SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_MANUAL, test_double)
{
	test_helper_features<uint8_t, double>(SG2CV_MANUAL);
	test_helper_features<int8_t, double>(SG2CV_MANUAL);
	test_helper_features<uint16_t, double>(SG2CV_MANUAL);
	test_helper_features<int16_t, double>(SG2CV_MANUAL);
	test_helper_features<int32_t, double>(SG2CV_MANUAL);
	test_helper_features<float32_t, double>(SG2CV_MANUAL);
	test_helper_features<float64_t, double>(SG2CV_MANUAL);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_8U)
{
	test_helper_features<uint8_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_8U, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_8U, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_unsigned_char)
{
	test_helper_features<uint8_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, unsigned char>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, unsigned char>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_8S)
{
	test_helper_features<uint8_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_8S, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_8S, SG2CV_CONSTRUCTOR);
}


TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_signed_char)
{
	test_helper_features<uint8_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, signed char>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, signed char>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_16U)
{
	test_helper_features<uint8_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_16U, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_16U, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_unsigned_short)
{
	test_helper_features<uint8_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, unsigned short>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, unsigned short>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_16S)
{
	test_helper_features<uint8_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_16S, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_16S, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_signed_short)
{
	test_helper_features<uint8_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, signed short>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, signed short>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_32S)
{
	test_helper_features<uint8_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_32S, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_32S, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_integer)
{
	test_helper_features<uint8_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, int>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, int>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_32F)
{
	test_helper_features<uint8_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_32F, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_32F, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_float)
{
	test_helper_features<uint8_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, float>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, float>(SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_CV_64F)
{
	test_helper_features<uint8_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t>(CV_64F, SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t>(CV_64F, SG2CV_CONSTRUCTOR);
}

TEST(DenseFeature2CVMat_using_SG2CV_CONSTRUCTOR, test_double)
{
	test_helper_features<uint8_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<int8_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<uint16_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<int16_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<int32_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<float32_t, double>(SG2CV_CONSTRUCTOR);
	test_helper_features<float64_t, double>(SG2CV_CONSTRUCTOR);
}
#endif //HAVE_OPENCV
