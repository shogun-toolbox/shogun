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
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/classifier/LDA.h>
#include <gtest/gtest.h>

using namespace shogun;
class DISABLED_LDATest: public::testing::Test
{
protected:

	//one setup for all tests.
	LDATest()
	{
		const int num=5;
		const int dims=3;
		const int classes=2;

		//Prepare the data for binary classification using LDA
		SGVector<float64_t> lab(classes*num);
		SGMatrix<float64_t> feat(dims, classes*num);

		feat(0, 0)=-2.81337943903340859;
		feat(0, 1)=-5.75992432468645088;
		feat(0, 2)=-5.4837468813610224;
		feat(0, 3)=-5.47479715849418014;
		feat(0, 4)=-4.39517634372092836;
		feat(0, 5)=-7.99471372898085164;
		feat(0, 6)=-7.10419958637667648;
		feat(0, 7)=-11.4850011467524826;
		feat(0, 8)=-9.67487852616068089;
		feat(0, 9)=-12.3194880071464681;

		feat(1, 0)=-6.56380697737734309;
		feat(1, 1)=-4.64057958297620488;
		feat(1, 2)=-3.59272857618575481;
		feat(1, 3)=-5.53059381299425468;
		feat(1, 4)=-5.16060076449381011;
		feat(1, 5)=11.0022309320050748;
		feat(1, 6)=10.4475375427292914;
		feat(1, 7)=10.3084005680058421;
		feat(1, 8)=10.3517589923186151;
		feat(1, 9)=11.1302375093709944;

		feat(2, 0)=-2.84357574928607937;
		feat(2, 1)=-6.40222758829360661;
		feat(2, 2)=-5.60120216303194862;
		feat(2, 3)=-4.99340782963709628;
		feat(2, 4)=-3.07109471770865472;
		feat(2, 5)=-11.3998413308604079;
		feat(2, 6)=-10.815184022595691;
		feat(2, 7)=-10.2965326042816976;
		feat(2, 8)=-9.49970263899488998;
		feat(2, 9)=-9.25703218159132923;

		for(int i=0; i<classes; ++i)
			for(int j=0; j<num; ++j)
				if(i==0)
					lab[i*num+j]=-1;
				else
					lab[i*num+j]=+1;

		labels=new CBinaryLabels(lab);
		features = new CDenseFeatures<float64_t>(feat);

		//Train the binary classifier.
		CLDA lda_FLD(0, features, labels, FLD_LDA);
		lda_FLD.train();

		//Test.
		CRegressionLabels* results_FLD=(lda_FLD.apply_regression(features));
		SG_REF(results_FLD);
		projection_FLD=results_FLD->get_labels();
		w_FLD = lda_FLD.get_w();
		SG_UNREF(results_FLD);

		//Train the binary classifier.
		CLDA lda_SVD(0, features, labels, SVD_LDA);
		lda_SVD.train();

		//Test.
		CRegressionLabels* results_SVD=lda_SVD.apply_regression(features);
		SG_REF(results_SVD);
		projection_SVD=results_SVD->get_labels();
		w_SVD=lda_SVD.get_w();
		SG_UNREF(results_SVD);
	}

	CBinaryLabels* labels;
	CDenseFeatures<float64_t>* features;
	SGVector<float64_t> projection_FLD;
	SGVector<float64_t> projection_SVD;
	SGVector<float64_t> w_FLD;
	SGVector<float64_t> w_SVD;
};

TEST_F(LDATest, CheckEigenvectors_FLD)
{
	// comparing our 'w' against 'w' a.k.a EigenVec of the scipy implementation
	// of Fisher 2 Class LDA here:
	// http://wiki.scipy.org/Cookbook/LinearClassification
	float64_t epsilon=0.00000001;
	EXPECT_NEAR(5.31296094, w_FLD[0], epsilon);
	EXPECT_NEAR(40.45747764, w_FLD[1], epsilon);
	EXPECT_NEAR(10.81046958, w_FLD[2], epsilon);
}

TEST_F(LDATest, DISABLED_CheckProjection_FLD)
{
	// No need of checking the binary labels if the following passes.
	float64_t epsilon=0.00000001;
	EXPECT_NEAR(-304.80621346, projection_FLD[0], epsilon);
	EXPECT_NEAR(-281.12285949, projection_FLD[1], epsilon);
	EXPECT_NEAR(-228.60266985, projection_FLD[2], epsilon);
	EXPECT_NEAR(-300.38571766, projection_FLD[3], epsilon);
	EXPECT_NEAR(-258.89964153, projection_FLD[4], epsilon);
	EXPECT_NEAR(+285.84589701, projection_FLD[5], epsilon);
	EXPECT_NEAR(+274.45608852, projection_FLD[6], epsilon);
	EXPECT_NEAR(+251.15879527, projection_FLD[7], epsilon);
	EXPECT_NEAR(+271.14418463, projection_FLD[8], epsilon);
	EXPECT_NEAR(+291.21213655, projection_FLD[9], epsilon);
}

TEST_F(LDATest, CheckEigenvectors_SVD)
{
	// comparing against the eigenvectors of the CanonVar implementation
	// in the brml toolbox, MATLAB.
	float64_t epsilon=0.00000001;
	EXPECT_NEAR(-0.09165651, w_SVD[0], epsilon);
	EXPECT_NEAR(+0.95744763, w_SVD[1], epsilon);
	EXPECT_NEAR(+0.27366605, w_SVD[2], epsilon);
}

TEST_F(LDATest, DISABLED_CheckProjection_SVD)
{
	//comparing agianst the projections from the CanonVar implementation
	//in the brml toolbox, MATLAB.
	float64_t epsilon=0.00000001;
	EXPECT_NEAR(-8.09643097, projection_SVD[0], epsilon);
	EXPECT_NEAR(-6.95885362, projection_SVD[1], epsilon);
	EXPECT_NEAR(-5.76169114, projection_SVD[2], epsilon);
	EXPECT_NEAR(-7.45158326, projection_SVD[3], epsilon);
	EXPECT_NEAR(-6.67021673, projection_SVD[4], epsilon);
	EXPECT_NEAR(+6.85547411, projection_SVD[5], epsilon);
	EXPECT_NEAR(+6.40276367, projection_SVD[6], epsilon);
	EXPECT_NEAR(+6.81301359, projection_SVD[7], epsilon);
	EXPECT_NEAR(+6.90668279, projection_SVD[8], epsilon);
	EXPECT_NEAR(+7.96084156, projection_SVD[9], epsilon);
}
