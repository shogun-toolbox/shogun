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

using namespace cv;
using namespace shogun;

TEST(CVMat2SGMatrixMemcpy, CVMat_to_SGMatrix_conversion_using_Memcpy)
{

	index_t data[] = {1, 3, 0,
					  0, 1, 0,
					  0, 0, 1};


	for (int l=0; l<7; l++)
    {
        Mat cvMat = Mat::eye(3,3,l);
	    
        switch (l)
        { 
            case 0:
        	{
        	    cvMat.at<uint8_t>(0,1) =3;
        	    break;
        	}
        
            case 1:
        	{
        	    cvMat.at<int8_t>(0,1) =3;
        	    break;
        	}
            case 2:
        	{
        	    cvMat.at<uint16_t>(0,1) =3;
        	    break;
        	}
            case 3:
        	{
        	    cvMat.at<int16_t>(0,1) =3;
        	    break;
        	}
            case 4:
        	{
        	    cvMat.at<int32_t>(0,1) =3;
        	    break;
        	}
            case 5:
        	{
        	    cvMat.at<float32_t>(0,1) =3;
        	    break;
        	}
            case 6:
        	{
        		cvMat.at<float64_t>(0,1) =3;
        		break;
        	}
    	}
	    
	    CDenseFeatures<uint8_t>* A1 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<int8_t>* A2 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<uint16_t>* A3 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<int16_t>* A4 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<int32_t>* A5 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<float32_t>* A6 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat,CV2SG_MEMCPY);
	    CDenseFeatures<float64_t>* A7 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat,CV2SG_MEMCPY);

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
}
#endif //HAVE_OPENCV
