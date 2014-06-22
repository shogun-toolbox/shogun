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
#include <shogun/lib/OpenCV/CV2SGMatrixFactory.h>
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

using namespace cv;
using namespace shogun;


template<typename CV_T, typename SG_T>void test_helper(CV2SGOptions option)
{
    index_t data[] =  {1, 3, 4, 0, 1, 0, 0, 0, 1};

    const int cv_type = OpenCVTypeName<CV_T>::get_opencv_type();

    Mat cvMat = Mat::eye(3,3,cv_type);
    cvMat.at<CV_T>(0,1) = 3;
    cvMat.at<CV_T>(0,2) = 4;

    SGMatrix<SG_T> sgMat = CV2SGMatrixFactory::getSGMatrix<SG_T>(cvMat, option);

    index_t k=0;
    for (index_t i=0; i< cvMat.rows; ++i)
    {
        for (index_t j=0; j< cvMat.cols; ++j)
        {
            EXPECT_EQ (sgMat(i,j),data[k]);
            ++k;
        }
    }
}

#define OPTION CV2SG_MEMCPY

TEST(CVMat2SGMatrix, test_uint8)
{
    test_helper<unsigned char,  uint8_t>(OPTION);
    test_helper<signed char,    uint8_t>(OPTION);
    test_helper<unsigned short, uint8_t>(OPTION);
    test_helper<signed short,   uint8_t>(OPTION);
    test_helper<int,            uint8_t>(OPTION);
    test_helper<float,          uint8_t>(OPTION);
    test_helper<double,         uint8_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_int8)
{
    test_helper<unsigned char,  int8_t>(OPTION);
    test_helper<signed char,    int8_t>(OPTION);
    test_helper<unsigned short, int8_t>(OPTION);
    test_helper<signed short,   int8_t>(OPTION);
    test_helper<int,            int8_t>(OPTION);
    test_helper<float,          int8_t>(OPTION);
    test_helper<double,         int8_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_uint16)
{
    test_helper<unsigned char,  uint16_t>(OPTION);
    test_helper<signed char,    uint16_t>(OPTION);
    test_helper<unsigned short, uint16_t>(OPTION);
    test_helper<signed short,   uint16_t>(OPTION);
    test_helper<int,            uint16_t>(OPTION);
    test_helper<float,          uint16_t>(OPTION);
    test_helper<double,         uint16_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_int16)
{
    test_helper<unsigned char,  int16_t>(OPTION);
    test_helper<signed char,    int16_t>(OPTION);
    test_helper<unsigned short, int16_t>(OPTION);
    test_helper<signed short,   int16_t>(OPTION);
    test_helper<int,            int16_t>(OPTION);
    test_helper<float,          int16_t>(OPTION);
    test_helper<double,         int16_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_int32)
{
    test_helper<unsigned char,  int32_t>(OPTION);
    test_helper<signed char,    int32_t>(OPTION);
    test_helper<unsigned short, int32_t>(OPTION);
    test_helper<signed short,   int32_t>(OPTION);
    test_helper<int,            int32_t>(OPTION);
    test_helper<float,          int32_t>(OPTION);
    test_helper<double,         int32_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_float32)
{
    test_helper<unsigned char,  float32_t>(OPTION);
    test_helper<signed char,    float32_t>(OPTION);
    test_helper<unsigned short, float32_t>(OPTION);
    test_helper<signed short,   float32_t>(OPTION);
    test_helper<int,            float32_t>(OPTION);
    test_helper<float,          float32_t>(OPTION);
    test_helper<double,         float32_t>(OPTION);
}

TEST(CVMat2SGMatrix, test_float64)
{
    test_helper<unsigned char,  float64_t>(OPTION);
    test_helper<signed char,    float64_t>(OPTION);
    test_helper<unsigned short, float64_t>(OPTION);
    test_helper<signed short,   float64_t>(OPTION);
    test_helper<int,            float64_t>(OPTION);
    test_helper<float,          float64_t>(OPTION);
    test_helper<double,         float64_t>(OPTION);
}

#endif //HAVE_OPENCV
