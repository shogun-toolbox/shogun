/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 Parijat Mazumdar
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

#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#ifdef HAVE_EIGEN3
#include <shogun/preprocessor/PCA.h>

using namespace shogun;

TEST(PCA, PCA_N_greater_D_EVD)
{
	SGMatrix<float64_t> data(3,5);
	data(0,0)=2.908008030729362;
	data(0,1)=-1.058180257987362;
	data(0,2)=1.098424617888623;
	data(0,3)=-2.051816299911149;
	data(0,4)=-1.577057022799202;
	data(1,0)=0.825218894228491;
	data(1,1)=-0.468615581100624;
	data(1,2)=-0.277871932787639;
	data(1,3)=-0.353849997774433;
	data(1,4)=0.507974650905946;
	data(2,0)=1.378971977916614;
	data(2,1)=-0.272469409250187;
	data(2,2)=0.701541458163284;
	data(2,3)=-0.823586525156853;
	data(2,4)=0.281984063670556;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(EVD);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.00000000000001;

	// comparing outputs against MATLAB 'princomp' implementation 
	EXPECT_NEAR(0.041855883987175,eigvec[0],epsilon);
	EXPECT_NEAR(0.291219269837891,eigvec[1],epsilon);
	EXPECT_NEAR(5.077526030285309,eigvec[2],epsilon);

	EXPECT_NEAR(0.238820512479407,transmat(0,0),epsilon);
	EXPECT_NEAR(-0.304406370622002,transmat(0,1),epsilon);
	EXPECT_NEAR(0.922117955764778,transmat(0,2),epsilon);
	EXPECT_NEAR(0.502124514814308,transmat(1,0),epsilon);
	EXPECT_NEAR(0.851501730295596,transmat(1,1),epsilon);
	EXPECT_NEAR(0.151048915673366,transmat(1,2),epsilon);
	EXPECT_NEAR(-0.831165287076865,transmat(2,0),epsilon);
	EXPECT_NEAR(0.426944451689378,transmat(2,1),epsilon);
	EXPECT_NEAR(0.356205980761254,transmat(2,2),epsilon);

	EXPECT_NEAR(0.182350122017013,finalmat(0,0),epsilon);
	EXPECT_NEAR(-0.041902251203685,finalmat(0,1),epsilon);
	EXPECT_NEAR(-0.240647729898028,finalmat(0,2),epsilon);
	EXPECT_NEAR(0.236493108746648,finalmat(0,3),epsilon);
	EXPECT_NEAR(-0.136293249661948,finalmat(0,4),epsilon);
	EXPECT_NEAR(0.216971008375464,finalmat(1,0),epsilon);
	EXPECT_NEAR(-0.382472041452699,finalmat(1,1),epsilon);
	EXPECT_NEAR(-0.460689222275080,finalmat(1,2),epsilon);
	EXPECT_NEAR(-0.217576202298234,finalmat(1,3),epsilon);
	EXPECT_NEAR(0.843766457650550,finalmat(1,4),epsilon);
	EXPECT_NEAR(3.325638119909419,finalmat(2,0),epsilon);
	EXPECT_NEAR(-1.115340910605008,finalmat(2,1),epsilon);
	EXPECT_NEAR(1.249063286478502,finalmat(2,2),epsilon);
	EXPECT_NEAR(-2.210566542225781,finalmat(2,3),epsilon);
	EXPECT_NEAR(-1.248793953557132,finalmat(2,4),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_N_equals_D_EVD)
{
	SGMatrix<float64_t> data(3,3);
	data(0,0)=2.908008030729362;
	data(0,1)=-1.058180257987362;
	data(0,2)=1.098424617888623;
	data(1,0)=0.825218894228491;
	data(1,1)=-0.468615581100624;
	data(1,2)=-0.277871932787639;
	data(2,0)=1.378971977916614;
	data(2,1)=-0.272469409250187;
	data(2,2)=0.701541458163284;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(EVD);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.00000001;

	// comparing outputs against MATLAB 'princomp' implementation 
	EXPECT_NEAR(0.0,eigvec[0],epsilon);
	EXPECT_NEAR(0.084750433,eigvec[1],epsilon);
	EXPECT_NEAR(5.03495863,eigvec[2],epsilon);

	EXPECT_NEAR(0.41770275,transmat(0,0),epsilon);
	EXPECT_NEAR(0.20781429,transmat(0,1),epsilon);
	EXPECT_NEAR(0.88449852,transmat(0,2),epsilon);
	EXPECT_NEAR(-0.13328384,transmat(1,0),epsilon);
	EXPECT_NEAR(-0.94894524,transmat(1,1),epsilon);
	EXPECT_NEAR(0.28589918,transmat(1,2),epsilon);
	EXPECT_NEAR(-0.8987546,transmat(2,0),epsilon);
	EXPECT_NEAR(0.23731023,transmat(2,1),epsilon);
	EXPECT_NEAR(0.36867875,transmat(2,2),epsilon);

	EXPECT_NEAR(0.0,finalmat(0,0),epsilon);
	EXPECT_NEAR(0.0,finalmat(0,1),epsilon);
	EXPECT_NEAR(0.0,finalmat(0,2),epsilon);
	EXPECT_NEAR(-0.173865951,finalmat(1,0),epsilon);
	EXPECT_NEAR(-0.162222411,finalmat(1,1),epsilon);
	EXPECT_NEAR(0.336088362,finalmat(1,2),epsilon);
	EXPECT_NEAR(2.21751537,finalmat(2,0),epsilon);
	EXPECT_NEAR(-2.26932988,finalmat(2,1),epsilon);
	EXPECT_NEAR(0.0518145101,finalmat(2,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_N_less_D_EVD)
{
	SGMatrix<float64_t> data(5,3);
	data(0,0)=0.033479882244451;
	data(0,1)=0.022889792751630;
	data(0,2)=-0.979206305167302;
	data(1,0)=-1.333677943428106;
	data(1,1)=-0.261995434966092;
	data(1,2)=-1.156401655664002;
	data(2,0)=1.127492278341590;
	data(2,1)=-1.750212368446790;
	data(2,2)=-0.533557109315987;
	data(3,0)=0.350179410603312;
	data(3,1)=-0.285650971595330;
	data(3,2)=-2.002635735883060;
	data(4,0)=-0.299066030332982;
	data(4,1)=-0.831366511567624;
	data(4,2)=0.964229422631627;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(EVD);
	pca->set_target_dim(2);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.00000000000001;

	// comparing outputs against MATLAB 'princomp' implementation
	EXPECT_NEAR(0,eigvec[0],epsilon);
	EXPECT_NEAR(2.327794822241147,eigvec[1],epsilon);
	EXPECT_NEAR(2.759160840481412,eigvec[2],epsilon);

	EXPECT_NEAR(0.258049566055304,transmat(0,0),epsilon);
	EXPECT_NEAR(0.257746561935451,transmat(0,1),epsilon);
	EXPECT_NEAR(0.349092719192590,transmat(1,0),epsilon);
	EXPECT_NEAR(-0.129544636386834,transmat(1,1),epsilon);
	EXPECT_NEAR(-0.630860251575450,transmat(2,0),epsilon);
	EXPECT_NEAR(0.648487498866225,transmat(2,1),epsilon);
	EXPECT_NEAR(0.374280965623520,transmat(3,0),epsilon);
	EXPECT_NEAR(0.647067522254220,transmat(3,1),epsilon);
	EXPECT_NEAR(-0.522947221638548,transmat(4,0),epsilon);
	EXPECT_NEAR(-0.278482463454826,transmat(4,1),epsilon);

	EXPECT_NEAR(-0.511467003751085,finalmat(0,0),epsilon);
	EXPECT_NEAR(1.715732114990145,finalmat(0,1),epsilon);
	EXPECT_NEAR(-1.204265111239059,finalmat(0,2),epsilon);
	EXPECT_NEAR(1.835430614937060,finalmat(1,0),epsilon);
	EXPECT_NEAR(-0.435473994643473,finalmat(1,1),epsilon);
	EXPECT_NEAR(-1.39995662029358,finalmat(1,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_N_greater_D_SVD)
{
	SGMatrix<float64_t> data(3,5);
	data(0,0)=2.908008030729362;
	data(0,1)=-1.058180257987362;
	data(0,2)=1.098424617888623;
	data(0,3)=-2.051816299911149;
	data(0,4)=-1.577057022799202;
	data(1,0)=0.825218894228491;
	data(1,1)=-0.468615581100624;
	data(1,2)=-0.277871932787639;
	data(1,3)=-0.353849997774433;
	data(1,4)=0.507974650905946;
	data(2,0)=1.378971977916614;
	data(2,1)=-0.272469409250187;
	data(2,2)=0.701541458163284;
	data(2,3)=-0.823586525156853;
	data(2,4)=0.281984063670556;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(SVD);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.0000000000001;

	// comparing outputs against BRMLtoolbox MATLAB 
	// http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Software
	EXPECT_NEAR(0.041855883987175,eigvec[2],epsilon);
	EXPECT_NEAR(0.291219269837891,eigvec[1],epsilon);
	EXPECT_NEAR(5.077526030285309,eigvec[0],epsilon);

	EXPECT_NEAR(-0.238820512479407,transmat(0,2),epsilon);
	EXPECT_NEAR(-0.304406370622002,transmat(0,1),epsilon);
	EXPECT_NEAR(0.922117955764778,transmat(0,0),epsilon);
	EXPECT_NEAR(-0.502124514814308,transmat(1,2),epsilon);
	EXPECT_NEAR(0.851501730295596,transmat(1,1),epsilon);
	EXPECT_NEAR(0.151048915673366,transmat(1,0),epsilon);
	EXPECT_NEAR(0.831165287076865,transmat(2,2),epsilon);
	EXPECT_NEAR(0.426944451689378,transmat(2,1),epsilon);
	EXPECT_NEAR(0.356205980761254,transmat(2,0),epsilon);

	EXPECT_NEAR(-0.182350122017013,finalmat(2,0),epsilon);
	EXPECT_NEAR(0.041902251203685,finalmat(2,1),epsilon);
	EXPECT_NEAR(0.240647729898028,finalmat(2,2),epsilon);
	EXPECT_NEAR(-0.236493108746648,finalmat(2,3),epsilon);
	EXPECT_NEAR(0.136293249661948,finalmat(2,4),epsilon);
	EXPECT_NEAR(0.216971008375464,finalmat(1,0),epsilon);
	EXPECT_NEAR(-0.382472041452699,finalmat(1,1),epsilon);
	EXPECT_NEAR(-0.460689222275080,finalmat(1,2),epsilon);
	EXPECT_NEAR(-0.217576202298234,finalmat(1,3),epsilon);
	EXPECT_NEAR(0.843766457650550,finalmat(1,4),epsilon);
	EXPECT_NEAR(3.325638119909419,finalmat(0,0),epsilon);
	EXPECT_NEAR(-1.115340910605008,finalmat(0,1),epsilon);
	EXPECT_NEAR(1.249063286478502,finalmat(0,2),epsilon);
	EXPECT_NEAR(-2.210566542225781,finalmat(0,3),epsilon);
	EXPECT_NEAR(-1.248793953557132,finalmat(0,4),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_N_equals_D_SVD)
{
	SGMatrix<float64_t> data(3,3);
	data(0,0)=2.908008030729362;
	data(0,1)=-1.058180257987362;
	data(0,2)=1.098424617888623;
	data(1,0)=0.825218894228491;
	data(1,1)=-0.468615581100624;
	data(1,2)=-0.277871932787639;
	data(2,0)=1.378971977916614;
	data(2,1)=-0.272469409250187;
	data(2,2)=0.701541458163284;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(SVD);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.00000001;

	// comparing outputs against MATLAB 'princomp' implementation 
	EXPECT_NEAR(5.03495863,eigvec[0],epsilon);
	EXPECT_NEAR(0.084750433,eigvec[1],epsilon);
	EXPECT_NEAR(0.0,eigvec[2],epsilon);

	EXPECT_NEAR(-0.41770275,transmat(0,2),epsilon);
	EXPECT_NEAR(-0.20781429,transmat(0,1),epsilon);
	EXPECT_NEAR(0.88449852,transmat(0,0),epsilon);
	EXPECT_NEAR(0.13328384,transmat(1,2),epsilon);
	EXPECT_NEAR(0.94894524,transmat(1,1),epsilon);
	EXPECT_NEAR(0.28589918,transmat(1,0),epsilon);
	EXPECT_NEAR(0.8987546,transmat(2,2),epsilon);
	EXPECT_NEAR(-0.23731023,transmat(2,1),epsilon);
	EXPECT_NEAR(0.36867875,transmat(2,0),epsilon);

	EXPECT_NEAR(0.0,finalmat(2,0),epsilon);
	EXPECT_NEAR(0.0,finalmat(2,1),epsilon);
	EXPECT_NEAR(0.0,finalmat(2,2),epsilon);
	EXPECT_NEAR(0.173865951,finalmat(1,0),epsilon);
	EXPECT_NEAR(0.162222411,finalmat(1,1),epsilon);
	EXPECT_NEAR(-0.336088362,finalmat(1,2),epsilon);
	EXPECT_NEAR(2.21751537,finalmat(0,0),epsilon);
	EXPECT_NEAR(-2.26932988,finalmat(0,1),epsilon);
	EXPECT_NEAR(0.0518145101,finalmat(0,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_N_less_D_SVD)
{
	SGMatrix<float64_t> data(5,3);
	data(0,0)=0.033479882244451;
	data(0,1)=0.022889792751630;
	data(0,2)=-0.979206305167302;
	data(1,0)=-1.333677943428106;
	data(1,1)=-0.261995434966092;
	data(1,2)=-1.156401655664002;
	data(2,0)=1.127492278341590;
	data(2,1)=-1.750212368446790;
	data(2,2)=-0.533557109315987;
	data(3,0)=0.350179410603312;
	data(3,1)=-0.285650971595330;
	data(3,2)=-2.002635735883060;
	data(4,0)=-0.299066030332982;
	data(4,1)=-0.831366511567624;
	data(4,2)=0.964229422631627;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(SVD);
	pca->set_target_dim(2);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.0000000000001;

	// comparing outputs against BRMLtoolbox MATLAB 
	// http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Software
	EXPECT_NEAR(0,eigvec[2],epsilon);
	EXPECT_NEAR(2.327794822241147,eigvec[1],epsilon);
	EXPECT_NEAR(2.759160840481412,eigvec[0],epsilon);

	EXPECT_NEAR(-0.258049566055304,transmat(0,1),epsilon);
	EXPECT_NEAR(-0.257746561935451,transmat(0,0),epsilon);
	EXPECT_NEAR(-0.349092719192590,transmat(1,1),epsilon);
	EXPECT_NEAR(0.129544636386834,transmat(1,0),epsilon);
	EXPECT_NEAR(0.630860251575450,transmat(2,1),epsilon);
	EXPECT_NEAR(-0.648487498866225,transmat(2,0),epsilon);
	EXPECT_NEAR(-0.374280965623520,transmat(3,1),epsilon);
	EXPECT_NEAR(-0.647067522254220,transmat(3,0),epsilon);
	EXPECT_NEAR(0.522947221638548,transmat(4,1),epsilon);
	EXPECT_NEAR(0.278482463454826,transmat(4,0),epsilon);

	EXPECT_NEAR(0.511467003751085,finalmat(1,0),epsilon);
	EXPECT_NEAR(-1.715732114990145,finalmat(1,1),epsilon);
	EXPECT_NEAR(1.204265111239059,finalmat(1,2),epsilon);
	EXPECT_NEAR(-1.835430614937060,finalmat(0,0),epsilon);
	EXPECT_NEAR(0.435473994643473,finalmat(0,1),epsilon);
	EXPECT_NEAR(1.39995662029358,finalmat(0,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_MEM_IN_PLACE)
{
	SGMatrix<float64_t> data(3,3);
	data(0,0)=2.908008030729362;
	data(0,1)=-1.058180257987362;
	data(0,2)=1.098424617888623;
	data(1,0)=0.825218894228491;
	data(1,1)=-0.468615581100624;
	data(1,2)=-0.277871932787639;
	data(2,0)=1.378971977916614;
	data(2,1)=-0.272469409250187;
	data(2,2)=0.701541458163284;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(AUTO,false,MEM_IN_PLACE);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> transmat=pca->get_transformation_matrix();
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);
	SGVector<float64_t> eigvec=pca->get_eigenvalues();

	float64_t epsilon = 0.00000001;

	// comparing outputs against MATLAB 'princomp' implementation 
	EXPECT_NEAR(5.03495863,eigvec[0],epsilon);
	EXPECT_NEAR(0.084750433,eigvec[1],epsilon);
	EXPECT_NEAR(0.0,eigvec[2],epsilon);

	EXPECT_NEAR(-0.41770275,transmat(0,2),epsilon);
	EXPECT_NEAR(-0.20781429,transmat(0,1),epsilon);
	EXPECT_NEAR(0.88449852,transmat(0,0),epsilon);
	EXPECT_NEAR(0.13328384,transmat(1,2),epsilon);
	EXPECT_NEAR(0.94894524,transmat(1,1),epsilon);
	EXPECT_NEAR(0.28589918,transmat(1,0),epsilon);
	EXPECT_NEAR(0.8987546,transmat(2,2),epsilon);
	EXPECT_NEAR(-0.23731023,transmat(2,1),epsilon);
	EXPECT_NEAR(0.36867875,transmat(2,0),epsilon);

	EXPECT_NEAR(0.0,finalmat(2,0),epsilon);
	EXPECT_NEAR(0.0,finalmat(2,1),epsilon);
	EXPECT_NEAR(0.0,finalmat(2,2),epsilon);
	EXPECT_NEAR(0.173865951,finalmat(1,0),epsilon);
	EXPECT_NEAR(0.162222411,finalmat(1,1),epsilon);
	EXPECT_NEAR(-0.336088362,finalmat(1,2),epsilon);
	EXPECT_NEAR(2.21751537,finalmat(0,0),epsilon);
	EXPECT_NEAR(-2.26932988,finalmat(0,1),epsilon);
	EXPECT_NEAR(0.0518145101,finalmat(0,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_apply_to_feature_vector_methodTest)
{
	SGMatrix<float64_t> data(5,3);
	data(0,0)=0.033479882244451;
	data(0,1)=0.022889792751630;
	data(0,2)=-0.979206305167302;
	data(1,0)=-1.333677943428106;
	data(1,1)=-0.261995434966092;
	data(1,2)=-1.156401655664002;
	data(2,0)=1.127492278341590;
	data(2,1)=-1.750212368446790;
	data(2,2)=-0.533557109315987;
	data(3,0)=0.350179410603312;
	data(3,1)=-0.285650971595330;
	data(3,2)=-2.002635735883060;
	data(4,0)=-0.299066030332982;
	data(4,1)=-0.831366511567624;
	data(4,2)=0.964229422631627;

	SGVector<float64_t> inputVector(5);
	inputVector[0] = 0.033479882244451;
	inputVector[1] = -1.333677943428106;
	inputVector[2] = 1.127492278341590;
	inputVector[3] = 0.350179410603312;
	inputVector[4] = -0.299066030332982;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(SVD);
	pca->set_target_dim(2);
	pca->init(features);

	SGVector<float64_t> finalVector=pca->apply_to_feature_vector(inputVector);

	// comparing outputs against BRMLtoolbox MATLAB 
	// http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.Software
	EXPECT_NEAR(-1.835430614937060, finalVector[0], 1e-13);
	EXPECT_NEAR(0.511467003751085, finalVector[1], 1e-13);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_WHITEN_SVD)
{
	SGMatrix<float64_t> data(5,3);
	data(0,0)=0.033479882244451;
	data(0,1)=0.022889792751630;
	data(0,2)=-0.979206305167302;
	data(1,0)=-1.333677943428106;
	data(1,1)=-0.261995434966092;
	data(1,2)=-1.156401655664002;
	data(2,0)=1.127492278341590;
	data(2,1)=-1.750212368446790;
	data(2,2)=-0.533557109315987;
	data(3,0)=0.350179410603312;
	data(3,1)=-0.285650971595330;
	data(3,2)=-2.002635735883060;
	data(4,0)=-0.299066030332982;
	data(4,1)=-0.831366511567624;
	data(4,2)=0.964229422631627;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(SVD,true);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);

	SGMatrix<float64_t> covariance_mat=finalmat.matrix_multiply(finalmat,finalmat,false,true);

	float64_t epsilon = 0.0000001;

	EXPECT_NEAR(1.0,covariance_mat(0,0),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(0,1),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(0,2),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(1,0),epsilon);
	EXPECT_NEAR(1.0,covariance_mat(1,1),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(1,2),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(2,0),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(2,1),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(2,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

TEST(PCA, PCA_WHITEN_EVD)
{
	SGMatrix<float64_t> data(5,3);
	data(0,0)=0.033479882244451;
	data(0,1)=0.022889792751630;
	data(0,2)=-0.979206305167302;
	data(1,0)=-1.333677943428106;
	data(1,1)=-0.261995434966092;
	data(1,2)=-1.156401655664002;
	data(2,0)=1.127492278341590;
	data(2,1)=-1.750212368446790;
	data(2,2)=-0.533557109315987;
	data(3,0)=0.350179410603312;
	data(3,1)=-0.285650971595330;
	data(3,2)=-2.002635735883060;
	data(4,0)=-0.299066030332982;
	data(4,1)=-0.831366511567624;
	data(4,2)=0.964229422631627;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	CPCA* pca=new CPCA(EVD,true);
	pca->set_target_dim(3);
	pca->init(features);

	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(features);

	SGMatrix<float64_t> covariance_mat=finalmat.matrix_multiply(finalmat,finalmat,false,true);

	float64_t epsilon = 0.0000001;

	EXPECT_NEAR(0.0,covariance_mat(0,0),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(0,1),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(0,2),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(1,0),epsilon);
	EXPECT_NEAR(1.0,covariance_mat(1,1),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(1,2),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(2,0),epsilon);
	EXPECT_NEAR(0.0,covariance_mat(2,1),epsilon);
	EXPECT_NEAR(1.0,covariance_mat(2,2),epsilon);

	SG_UNREF(pca);
	SG_UNREF(features);
}

#endif //HAVE_EIGEN3
