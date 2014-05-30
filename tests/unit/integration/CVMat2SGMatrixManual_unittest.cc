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

TEST(CVMat2SGMatrixManual, CVMat_to_SGMatrix_conversion_using_Manual)
{

	index_t data[] = {1, 3, 0,
					  0, 1, 0,
					  0, 0, 1};


	Mat cvMat2 = Mat::eye(3,3,CV_8U);
	cvMat2.at<unsigned char>(0,1) =3;


	CDenseFeatures<uint8_t>* A11 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A12 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A13 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A14 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A15 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A16 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat2,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A17 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat2,CV2SG_MANUAL);

	SGMatrix<uint8_t> B11 = A11->get_feature_matrix();
    SGMatrix<int8_t> B12 = A12->get_feature_matrix();
	SGMatrix<uint16_t> B13 = A13->get_feature_matrix();
	SGMatrix<int16_t> B14 = A14->get_feature_matrix();
	SGMatrix<int32_t> B15 = A15->get_feature_matrix();
	SGMatrix<float32_t> B16 = A16->get_feature_matrix();
	SGMatrix<float64_t> B17 = A17->get_feature_matrix();


	Mat cvMat3 = Mat::eye(3,3,CV_8S);
	cvMat3.at<char>(0,1)=3;

	CDenseFeatures<uint8_t>* A21 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A22 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A23 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A24 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A25 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A26 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat3,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A27 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat3,CV2SG_MANUAL);

	SGMatrix<uint8_t> B21 = A21->get_feature_matrix();
	SGMatrix<int8_t> B22 = A22->get_feature_matrix();
	SGMatrix<uint16_t> B23 = A23->get_feature_matrix();
	SGMatrix<int16_t> B24 = A24->get_feature_matrix();
	SGMatrix<int32_t> B25 = A25->get_feature_matrix();
	SGMatrix<float32_t> B26 = A26->get_feature_matrix();
	SGMatrix<float64_t> B27 = A27->get_feature_matrix();


	Mat cvMat4 = Mat::eye(3,3,CV_16U);
	cvMat4.at<ushort>(0,1)=3;

	CDenseFeatures<uint8_t>* A31 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A32 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A33 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A34 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A35 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A36 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat4,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A37 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat4,CV2SG_MANUAL);

	SGMatrix<uint8_t> B31 = A31->get_feature_matrix();
	SGMatrix<int8_t> B32 = A32->get_feature_matrix();
	SGMatrix<uint16_t> B33 = A33->get_feature_matrix();
	SGMatrix<int16_t> B34 = A34->get_feature_matrix();
	SGMatrix<int32_t> B35 = A35->get_feature_matrix();
	SGMatrix<float32_t> B36 = A36->get_feature_matrix();
	SGMatrix<float64_t> B37 = A37->get_feature_matrix();


	Mat cvMat5 = Mat::eye(3,3,CV_16S);
	cvMat5.at<short>(0,1)=3;

	CDenseFeatures<uint8_t>* A41 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A42 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A43 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A44 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A45 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A46 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat5,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A47 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat5,CV2SG_MANUAL);

	SGMatrix<uint8_t> B41 = A41->get_feature_matrix();
	SGMatrix<int8_t> B42 = A42->get_feature_matrix();
	SGMatrix<uint16_t> B43 = A43->get_feature_matrix();
	SGMatrix<int16_t> B44 = A44->get_feature_matrix();
	SGMatrix<int32_t> B45 = A45->get_feature_matrix();
	SGMatrix<float32_t> B46 = A46->get_feature_matrix();
	SGMatrix<float64_t> B47 = A47->get_feature_matrix();


	Mat cvMat6 = Mat::eye(3,3,CV_32S);
	cvMat6.at<int>(0,1)=3;

	CDenseFeatures<uint8_t>* A51 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A52 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A53 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A54 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A55 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A56 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat6,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A57 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat6,CV2SG_MANUAL);

	SGMatrix<uint8_t> B51 = A51->get_feature_matrix();
	SGMatrix<int8_t> B52 = A52->get_feature_matrix();
	SGMatrix<uint16_t> B53 = A53->get_feature_matrix();
	SGMatrix<int16_t> B54 = A54->get_feature_matrix();
	SGMatrix<int32_t> B55 = A55->get_feature_matrix();
	SGMatrix<float32_t> B56 = A56->get_feature_matrix();
	SGMatrix<float64_t> B57 = A57->get_feature_matrix();


	Mat cvMat7 = Mat::eye(3,3,CV_32F);
	cvMat7.at<float>(0,1)=3;

	CDenseFeatures<uint8_t>* A61 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A62 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A63 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A64 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A65 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A66 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat7,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A67 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat7,CV2SG_MANUAL);

	SGMatrix<uint8_t> B61 = A61->get_feature_matrix();
	SGMatrix<int8_t> B62 = A62->get_feature_matrix();
	SGMatrix<uint16_t> B63 = A63->get_feature_matrix();
	SGMatrix<int16_t> B64 = A64->get_feature_matrix();
	SGMatrix<int32_t> B65 = A65->get_feature_matrix();
	SGMatrix<float32_t> B66 = A66->get_feature_matrix();
	SGMatrix<float64_t> B67 = A67->get_feature_matrix();

	Mat cvMat1 = Mat::eye(3,3,CV_64F);
	cvMat1.at<double>(0,1)=3;


	CDenseFeatures<uint8_t>* A71 = CV2FeaturesFactory::getDenseFeatures<uint8_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<int8_t>* A72 = CV2FeaturesFactory::getDenseFeatures<int8_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<uint16_t>* A73 = CV2FeaturesFactory::getDenseFeatures<uint16_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<int16_t>* A74 = CV2FeaturesFactory::getDenseFeatures<int16_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<int32_t>* A75 = CV2FeaturesFactory::getDenseFeatures<int32_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<float32_t>* A76 = CV2FeaturesFactory::getDenseFeatures<float32_t>(cvMat1,CV2SG_MANUAL);
	CDenseFeatures<float64_t>* A77 = CV2FeaturesFactory::getDenseFeatures<float64_t>(cvMat1,CV2SG_MANUAL);

	SGMatrix<uint8_t> B71 = A71->get_feature_matrix();
	SGMatrix<int8_t> B72 = A72->get_feature_matrix();
	SGMatrix<uint16_t> B73 = A73->get_feature_matrix();
	SGMatrix<int16_t> B74 = A74->get_feature_matrix();
	SGMatrix<int32_t> B75 = A75->get_feature_matrix();
	SGMatrix<float32_t> B76 = A76->get_feature_matrix();
	SGMatrix<float64_t> B77 = A77->get_feature_matrix();


	index_t k=0;
	for (index_t i=0; i< 3; ++i)
	{
		for (index_t j=0; j< 3; ++j)
		{
			EXPECT_EQ (B11(i,j),data[k]);
			EXPECT_EQ (B12(i,j),data[k]);
			EXPECT_EQ (B13(i,j),data[k]);
			EXPECT_EQ (B14(i,j),data[k]);
			EXPECT_EQ (B15(i,j),data[k]);
			EXPECT_EQ (B16(i,j),data[k]);
			EXPECT_EQ (B17(i,j),data[k]);

			EXPECT_EQ (B21(i,j),data[k]);
			EXPECT_EQ (B22(i,j),data[k]);
			EXPECT_EQ (B23(i,j),data[k]);
			EXPECT_EQ (B24(i,j),data[k]);
			EXPECT_EQ (B25(i,j),data[k]);
			EXPECT_EQ (B26(i,j),data[k]);
			EXPECT_EQ (B27(i,j),data[k]);

			EXPECT_EQ (B31(i,j),data[k]);
			EXPECT_EQ (B32(i,j),data[k]);
			EXPECT_EQ (B33(i,j),data[k]);
			EXPECT_EQ (B34(i,j),data[k]);
			EXPECT_EQ (B35(i,j),data[k]);
			EXPECT_EQ (B36(i,j),data[k]);
			EXPECT_EQ (B37(i,j),data[k]);

			EXPECT_EQ (B41(i,j),data[k]);
			EXPECT_EQ (B42(i,j),data[k]);
			EXPECT_EQ (B43(i,j),data[k]);
			EXPECT_EQ (B44(i,j),data[k]);
			EXPECT_EQ (B45(i,j),data[k]);
			EXPECT_EQ (B46(i,j),data[k]);
			EXPECT_EQ (B47(i,j),data[k]);

			EXPECT_EQ (B51(i,j),data[k]);
			EXPECT_EQ (B52(i,j),data[k]);
			EXPECT_EQ (B53(i,j),data[k]);
			EXPECT_EQ (B54(i,j),data[k]);
			EXPECT_EQ (B55(i,j),data[k]);
			EXPECT_EQ (B56(i,j),data[k]);
			EXPECT_EQ (B57(i,j),data[k]);

			EXPECT_EQ (B61(i,j),data[k]);
			EXPECT_EQ (B62(i,j),data[k]);
			EXPECT_EQ (B63(i,j),data[k]);
			EXPECT_EQ (B64(i,j),data[k]);
			EXPECT_EQ (B65(i,j),data[k]);
			EXPECT_EQ (B66(i,j),data[k]);
			EXPECT_EQ (B67(i,j),data[k]);

			EXPECT_EQ (B71(i,j),data[k]);
			EXPECT_EQ (B72(i,j),data[k]);
			EXPECT_EQ (B73(i,j),data[k]);
			EXPECT_EQ (B74(i,j),data[k]);
			EXPECT_EQ (B75(i,j),data[k]);
			EXPECT_EQ (B76(i,j),data[k]);
			EXPECT_EQ (B77(i,j),data[k]);
			++k;
		}
	}
}
#endif //HAVE_OPENCV

