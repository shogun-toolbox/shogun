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
#include <shogun/lib/OpenCV/OpenCVTypeName.h>

using namespace cv;
using namespace shogun;

class Test_Helper_Class
{   
    public:        
    template <typename T> static void test_Helper_Function(CV2SGOptions);
};

    template<typename T>void Test_Helper_Class::test_Helper_Function(CV2SGOptions option)
    {
        index_t data[] =  {1, 3, 4, 0, 1, 0, 0, 0, 1};

        const int test_type = OpenCVTypeName<T>::get_opencv_type();
        Mat cvMat = Mat::eye(3,3,test_type);
        cvMat.at<T>(0,1) = 3;
        cvMat.at<T>(0,2) = 4;
   
        CDenseFeatures<uint8_t>* A1 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat,option);
	    CDenseFeatures<int8_t>* A2 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat,option);
	    CDenseFeatures<uint16_t>* A3 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat,option);
	    CDenseFeatures<int16_t>* A4 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat,option);
	    CDenseFeatures<int32_t>* A5 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat,option);
	    CDenseFeatures<float32_t>* A6 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat,option);
	    CDenseFeatures<float64_t>* A7 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat,option);

	    SGMatrix<uint8_t> B1 = A1->get_feature_matrix();
        SGMatrix<int8_t> B2 = A2->get_feature_matrix();
	    SGMatrix<uint16_t> B3 = A3->get_feature_matrix();
	    SGMatrix<int16_t> B4 = A4->get_feature_matrix();
	    SGMatrix<int32_t> B5 = A5->get_feature_matrix();
	    SGMatrix<float32_t> B6 = A6->get_feature_matrix();
	    SGMatrix<float64_t> B7 = A7->get_feature_matrix();

   	    index_t k=0;	  
        for (index_t i=0; i< cvMat.rows; ++i)
	    {
		    for (index_t j=0; j< cvMat.cols; ++j)
		    {
			    EXPECT_EQ (B1(i,j),data[k]);
                EXPECT_EQ (B2(i,j),data[k]);
			    EXPECT_EQ (B3(i,j),data[k]);
			    EXPECT_EQ (B4(i,j),data[k]);
			    EXPECT_EQ (B5(i,j),data[k]);
			    EXPECT_EQ (B6(i,j),data[k]);
			    EXPECT_EQ (B7(i,j),data[k]);
                ++k;
            }
        }
    }

TEST(CVMat2SGMatrixManual, CVMat_to_SGMatrix_conversion_using_Manual)
{    
    Test_Helper_Class::test_Helper_Function<uint8_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<int8_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<uint16_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<int16_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<int32_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<float32_t>(CV2SG_MANUAL);
    Test_Helper_Class::test_Helper_Function<float64_t>(CV2SG_MANUAL); 
}

TEST(CVMat2SGMatrixMemcpy, CVMat_to_SGMatrix_conversion_using_Memcpy)
{    
    Test_Helper_Class::test_Helper_Function<uint8_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int8_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<uint16_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int16_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<int32_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<float32_t>(CV2SG_MEMCPY);
    Test_Helper_Class::test_Helper_Function<float64_t>(CV2SG_MEMCPY); 
}
#endif //HAVE_OPENCV
