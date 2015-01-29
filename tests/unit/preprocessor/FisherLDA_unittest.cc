/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 Abhijeet Kislay
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <gtest/gtest.h>
#include <shogun/lib/common.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include<shogun/preprocessor/FisherLDA.h>

using namespace shogun;

class FLDATest: public::testing::Test
{

protected:

	FLDATest()
	{
		const int num=6;
		const int dims=4;
		const int classes=2;

		SGMatrix<float64_t> test_matrix(dims,classes*num);
		SGVector<float64_t> labels_vector(classes*num);
		//prepare the data for dimensional reduction using LDA on the following 
		//excerpt from the Digit database.

		test_matrix(0,0)=50;   
		test_matrix(1,0)=121;
		test_matrix(2,0)=215;  
		test_matrix(3,0)=254; 

		test_matrix(0,1)=241;
		test_matrix(1,1)=254;
		test_matrix(2,1)=253;
		test_matrix(3,1)=253;

		test_matrix(0,2)=252;
		test_matrix(1,2)=252;
		test_matrix(2,2)=252;
		test_matrix(3,2)=252;

		test_matrix(0,3)=0;
		test_matrix(1,3)=16;
		test_matrix(2,3)=161;
		test_matrix(3,3)=232;

		test_matrix(0,4)=0;
		test_matrix(1,4)=0;
		test_matrix(2,4)=0;
		test_matrix(3,4)=0;

		test_matrix(0,5)=0;
		test_matrix(1,5)=0;
		test_matrix(2,5)=157;
		test_matrix(3,5)=252;

		test_matrix(0,6)=105;
		test_matrix(1,6)=254;
		test_matrix(2,6)=229;
		test_matrix(3,6)=24;

		test_matrix(0,7)=163;
		test_matrix(1,7)=0;
		test_matrix(2,7)=0;
		test_matrix(3,7)=0;

		test_matrix(0,8)=253;
		test_matrix(1,8)=234;
		test_matrix(2,8)=253;
		test_matrix(3,8)=253;

		test_matrix(0,9)=249;
		test_matrix(1,9)=249;
		test_matrix(2,9)=238;
		test_matrix(3,9)=168;

		test_matrix(0,10)=109;
		test_matrix(1,10)=109;
		test_matrix(2,10)=191;
		test_matrix(3,10)=255;

		test_matrix(0,11)=253;
		test_matrix(1,11)=252;
		test_matrix(2,11)=186;
		test_matrix(3,11)=56;

		dense_feat=new CDenseFeatures<float64_t>(test_matrix);
		for(int i=0; i<classes; ++i)
			for(int j=0; j<num; ++j)
					labels_vector[i*num+j]=i;
		labels=new CMulticlassLabels(labels_vector);
	
	}
	CDenseFeatures<float64_t>* dense_feat;
	CMulticlassLabels* labels;
};



TEST_F(FLDATest, CANVAR_FLDA_Unit_test)
{

	SG_REF(dense_feat);
	SG_REF(labels);

	// comparing outputs against BRMLtoolbox MATLAB "CannonVar.m" implementation 
	// http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Software
	CFisherLDA fisherlda(CANVAR_FLDA);
	fisherlda.fit(dense_feat, labels, 1);
	SGMatrix<float64_t> y=fisherlda.apply_to_feature_matrix(dense_feat);
   
	float64_t epsilon=0.00000000001;

	// comparing projection outputs against Matlab 'CannonVar' implementation
	EXPECT_NEAR(+51.3760082238439111, y(0,0), epsilon); 
	EXPECT_NEAR(-56.4869601858493127, y(0,1), epsilon); 
	EXPECT_NEAR(-63.3150724246548862, y(0,2), epsilon); 
	EXPECT_NEAR(+75.1693851132601338, y(0,3), epsilon); 
	EXPECT_NEAR( 0.00000000000000000, y(0,4), epsilon); 
	EXPECT_NEAR(+87.4243337392790067, y(0,5), epsilon); 
	EXPECT_NEAR(-114.092258182259556, y(0,6), epsilon); 
	EXPECT_NEAR(-94.010102920274619,  y(0,7), epsilon); 
	EXPECT_NEAR(-66.5754164410987244, y(0,8), epsilon); 
	EXPECT_NEAR(-109.483855254332568, y(0,9), epsilon); 
	EXPECT_NEAR(+27.5378037199449182, y(0,10), epsilon); 
	EXPECT_NEAR(-158.649815592855759, y(0,11), epsilon); 
   
	// comparing eigenvectors from the transformation_matrix with that from the
	// 'CannonVar' implementation.
	SGMatrix<float64_t> transformy=fisherlda.get_transformation_matrix(); 
	EXPECT_NEAR(-0.576749097670396393, transformy[0], epsilon); 
	EXPECT_NEAR(+0.158373354160231516, transformy[1], epsilon); 
	EXPECT_NEAR(-0.47693388209865617, transformy[2], epsilon); 
	EXPECT_NEAR(+0.64405933820939687, transformy[3], epsilon); 

	SG_UNREF(dense_feat);
	SG_UNREF(labels);
}

TEST_F(FLDATest, CLASSIC_FLDA_Unit_test)
{
    
	SG_REF(dense_feat);
	SG_REF(labels);

	CFisherLDA fisherlda(CLASSIC_FLDA);
	fisherlda.fit(dense_feat, labels, 1);
	SGMatrix<float64_t> y=fisherlda.apply_to_feature_matrix(dense_feat);
   
	float64_t epsilon=0.00000000001;

	// comparing projection outputs against OpenCV's LDA implementation
	EXPECT_NEAR(-2.320638860636734, y(0,0), epsilon); 
	EXPECT_NEAR(-17.1971943875009,	y(0,1), epsilon); 
	EXPECT_NEAR(-21.37035944868713, y(0,2), epsilon); 
	EXPECT_NEAR(-14.00799864614085, y(0,3), epsilon); 
	EXPECT_NEAR( 0.000000000000000, y(0,4), epsilon); 
	EXPECT_NEAR(-12.17552228293873, y(0,5), epsilon); 
	EXPECT_NEAR(-45.63707656856615, y(0,6), epsilon); 
	EXPECT_NEAR(-50.59105059954445, y(0,7), epsilon); 
	EXPECT_NEAR( -31.59115995427943,y(0,8), epsilon); 
	EXPECT_NEAR(-44.37699306363105, y(0,9), epsilon); 
	EXPECT_NEAR(-10.12764069879628, y(0,10),epsilon); 
	EXPECT_NEAR(-50.84827819267112, y(0,11),epsilon); 
   
	// comparing eigenvectors from the transformation_matrix with that from the
	// 'opencv LDA' implementation.
	SGMatrix<float64_t> transformy=fisherlda.get_transformation_matrix(); 
	EXPECT_NEAR(-0.3103745435554874, transformy[0], epsilon); 
	EXPECT_NEAR(+0.5334735522056344, transformy[1], epsilon); 
	EXPECT_NEAR(-0.6885872352166886, transformy[2], epsilon); 
	EXPECT_NEAR(+0.3806852128812753, transformy[3], epsilon); 

	SG_UNREF(dense_feat);
	SG_UNREF(labels);
}


TEST(FLDATesti, CANVAR_FLDA_for_D_greater_than_N )
{ 
	SGMatrix<float64_t> test_matrix(6,5);
	SGVector<float64_t> labels_vector(5);

	test_matrix(0,0)=50.6060;
	test_matrix(1,0)=11.3334;
	test_matrix(2,0)=36.3943;  
	test_matrix(3,0)=53.9095; 
	test_matrix(4,0)=47.2621; 
	test_matrix(5,0)=40.0941; 

	test_matrix(0,1)=53.6064;
	test_matrix(1,1)=38.0054;
	test_matrix(2,1)=31.1771;  
	test_matrix(3,1)=65.3217; 
	test_matrix(4,1)=73.6402; 
	test_matrix(5,1)=93.0575; 

	test_matrix(0,2)=87.8634;
	test_matrix(1,2)=25.5605;
	test_matrix(2,2)=67.2183;  
	test_matrix(3,2)=96.8582; 
	test_matrix(4,2)=85.6729; 
	test_matrix(5,2)=85.5027; 

	test_matrix(0,3)=82.4306;
	test_matrix(1,3)=41.2554;
	test_matrix(2,3)=51.4481;  
	test_matrix(3,3)=94.6800; 
	test_matrix(4,3)=98.0661; 
	test_matrix(5,3)=108.5917; 

	test_matrix(0,4)=109.0127; 
	test_matrix(1,4)=30.9425;
	test_matrix(2,4)=77.7320;  
	test_matrix(3,4)=118.4820; 
	test_matrix(4,4)=107.3717; 
	test_matrix(5,4)=99.9955; 

	labels_vector[0]=0;
	labels_vector[1]=0;
	labels_vector[2]=0;
	labels_vector[3]=1;
	labels_vector[4]=1;
	
	CMulticlassLabels* labels=new CMulticlassLabels(labels_vector);
	CDenseFeatures<float64_t>* dense_feat=new CDenseFeatures<float64_t>(test_matrix);
	
	SG_REF(labels);
	SG_REF(dense_feat);

	CFisherLDA fisherlda(CANVAR_FLDA);
	fisherlda.fit(dense_feat, labels, 1);
	SGMatrix<float64_t> transformy=fisherlda.get_transformation_matrix();  
 
	// comparing eigenvectors from the transformation_matrix with that from the
	// 'CannonVar' implementation. 
	float64_t epsilon=0.00000000001; 
	EXPECT_NEAR(-0.338514731928807433, transformy[0], epsilon); 
	EXPECT_NEAR(-0.106942313169695741, transformy[1], epsilon); 
	EXPECT_NEAR( 0.61367409029250708,  transformy[2], epsilon); 
	EXPECT_NEAR(-0.162039434021644224, transformy[3], epsilon); 
	EXPECT_NEAR(-0.600331522116284155, transformy[4], epsilon); 
	EXPECT_NEAR( 0.332746922149909308, transformy[5], epsilon); 

	SG_UNREF(labels);
	SG_UNREF(dense_feat);
}
#endif//HAVE_EIGEN3
