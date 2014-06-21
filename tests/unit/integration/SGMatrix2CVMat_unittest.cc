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


template<typename SG_T, typename CV_T>void test_helper(SG2CVOptions option)
{
    index_t expected[] = {1, 3, 4, 0, 1, 0, 0, 0, 1};

    SGMatrix<SG_T> sgMat = SGMatrix<SG_T>::create_identity_matrix(3, 1);
    sgMat(0,1) = 3;
    sgMat(0,2) = 4;

    Mat cvMat = SG2CVMatFactory::getcvMat<SG_T,CV_T>(sgMat, option);

    index_t k=0;
    for (index_t i=0; i<3; ++i)
    {
        for (index_t j=0; j<3; ++j)
        {
            EXPECT_EQ(cvMat.at< CV_T >(i,j), expected[k]);
            ++k;
        }
    }
}

template<typename SG_T>void test_helper(int cv_type, SG2CVOptions option)
{
    index_t expected[] = {1, 3, 4, 0, 1, 0, 0, 0, 1};

    SGMatrix<SG_T> sgMat = SGMatrix<SG_T>::create_identity_matrix(3, 1);
    sgMat(0,1) = 3;
    sgMat(0,2) = 4;

    Mat cvMat = SG2CVMatFactory::getcvMat<SG_T>(sgMat, cv_type, option);

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

#define OPTION SG2CV_MEMCPY

TEST(SGMatrix2CVMat, test_CV_8U)
{
    test_helper<uint8_t  >(CV_8U, OPTION);
    test_helper<int8_t   >(CV_8U, OPTION);
    test_helper<uint16_t >(CV_8U, OPTION);
    test_helper<int16_t  >(CV_8U, OPTION);
    test_helper<int32_t  >(CV_8U, OPTION);
    test_helper<float32_t>(CV_8U, OPTION);
    test_helper<float64_t>(CV_8U, OPTION);
}

TEST(SGMatrix2CVMat, test_unsigned_char)
{
    test_helper<uint8_t,   unsigned char>(OPTION);
    test_helper<int8_t,    unsigned char>(OPTION);
    test_helper<uint16_t,  unsigned char>(OPTION);
    test_helper<int16_t,   unsigned char>(OPTION);
    test_helper<int32_t,   unsigned char>(OPTION);
    test_helper<float32_t, unsigned char>(OPTION);
    test_helper<float64_t, unsigned char>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_8S)
{
    test_helper<uint8_t  >(CV_8S, OPTION);
    test_helper<int8_t   >(CV_8S, OPTION);
    test_helper<uint16_t >(CV_8S, OPTION);
    test_helper<int16_t  >(CV_8S, OPTION);
    test_helper<int32_t  >(CV_8S, OPTION);
    test_helper<float32_t>(CV_8S, OPTION);
    test_helper<float64_t>(CV_8S, OPTION);
}


TEST(SGMatrix2CVMat, test_signed_char)
{
    test_helper<uint8_t,   signed char>(OPTION);
    test_helper<int8_t,    signed char>(OPTION);
    test_helper<uint16_t,  signed char>(OPTION);
    test_helper<int16_t,   signed char>(OPTION);
    test_helper<int32_t,   signed char>(OPTION);
    test_helper<float32_t, signed char>(OPTION);
    test_helper<float64_t, signed char>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_16U)
{
    test_helper<uint8_t  >(CV_16U, OPTION);
    test_helper<int8_t   >(CV_16U, OPTION);
    test_helper<uint16_t >(CV_16U, OPTION);
    test_helper<int16_t  >(CV_16U, OPTION);
    test_helper<int32_t  >(CV_16U, OPTION);
    test_helper<float32_t>(CV_16U, OPTION);
    test_helper<float64_t>(CV_16U, OPTION);
}

TEST(SGMatrix2CVMat, test_unsigned_short)
{
    test_helper<uint8_t,   unsigned short>(OPTION);
    test_helper<int8_t,    unsigned short>(OPTION);
    test_helper<uint16_t,  unsigned short>(OPTION);
    test_helper<int16_t,   unsigned short>(OPTION);
    test_helper<int32_t,   unsigned short>(OPTION);
    test_helper<float32_t, unsigned short>(OPTION);
    test_helper<float64_t, unsigned short>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_16S)
{
    test_helper<uint8_t  >(CV_16S, OPTION);
    test_helper<int8_t   >(CV_16S, OPTION);
    test_helper<uint16_t >(CV_16S, OPTION);
    test_helper<int16_t  >(CV_16S, OPTION);
    test_helper<int32_t  >(CV_16S, OPTION);
    test_helper<float32_t>(CV_16S, OPTION);
    test_helper<float64_t>(CV_16S, OPTION);
}

TEST(SGMatrix2CVMat, test_signed_short)
{
    test_helper<uint8_t,   signed short>(OPTION);
    test_helper<int8_t,    signed short>(OPTION);
    test_helper<uint16_t,  signed short>(OPTION);
    test_helper<int16_t,   signed short>(OPTION);
    test_helper<int32_t,   signed short>(OPTION);
    test_helper<float32_t, signed short>(OPTION);
    test_helper<float64_t, signed short>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_32S)
{
    test_helper<uint8_t  >(CV_32S, OPTION);
    test_helper<int8_t   >(CV_32S, OPTION);
    test_helper<uint16_t >(CV_32S, OPTION);
    test_helper<int16_t  >(CV_32S, OPTION);
    test_helper<int32_t  >(CV_32S, OPTION);
    test_helper<float32_t>(CV_32S, OPTION);
    test_helper<float64_t>(CV_32S, OPTION);
}

TEST(SGMatrix2CVMat, test_integer)
{
    test_helper<uint8_t,   int>(OPTION);
    test_helper<int8_t,    int>(OPTION);
    test_helper<uint16_t,  int>(OPTION);
    test_helper<int16_t,   int>(OPTION);
    test_helper<int32_t,   int>(OPTION);
    test_helper<float32_t, int>(OPTION);
    test_helper<float64_t, int>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_32F)
{
    test_helper<uint8_t  >(CV_32F, OPTION);
    test_helper<int8_t   >(CV_32F, OPTION);
    test_helper<uint16_t >(CV_32F, OPTION);
    test_helper<int16_t  >(CV_32F, OPTION);
    test_helper<int32_t  >(CV_32F, OPTION);
    test_helper<float32_t>(CV_32F, OPTION);
    test_helper<float64_t>(CV_32F, OPTION);
}

TEST(SGMatrix2CVMat, test_float)
{
    test_helper<uint8_t,   float>(OPTION);
    test_helper<int8_t,    float>(OPTION);
    test_helper<uint16_t,  float>(OPTION);
    test_helper<int16_t,   float>(OPTION);
    test_helper<int32_t,   float>(OPTION);
    test_helper<float32_t, float>(OPTION);
    test_helper<float64_t, float>(OPTION);
}

TEST(SGMatrix2CVMat, test_CV_64F)
{
    test_helper<uint8_t  >(CV_64F, OPTION);
    test_helper<int8_t   >(CV_64F, OPTION);
    test_helper<uint16_t >(CV_64F, OPTION);
    test_helper<int16_t  >(CV_64F, OPTION);
    test_helper<int32_t  >(CV_64F, OPTION);
    test_helper<float32_t>(CV_64F, OPTION);
    test_helper<float64_t>(CV_64F, OPTION);
}

TEST(SGMatrix2CVMat, test_double)
{
    test_helper<uint8_t,   double>(OPTION);
    test_helper<int8_t,    double>(OPTION);
    test_helper<uint16_t,  double>(OPTION);
    test_helper<int16_t,   double>(OPTION);
    test_helper<int32_t,   double>(OPTION);
    test_helper<float32_t, double>(OPTION);
    test_helper<float64_t, double>(OPTION);
}
#endif //HAVE_OPENCV
