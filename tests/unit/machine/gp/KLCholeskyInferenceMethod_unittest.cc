/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 *
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/machine/gp/ProbitVGLikelihood.h>
#include <shogun/machine/gp/StudentsTVGLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

TEST(KLCholeskyInferenceMethod,get_cholesky_t_likelihood)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTVGLikelihood* likelihood=new CStudentsTVGLikelihood(1, 3);

	// specify GP regression with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// L =
	//
	//-0.582190787031602   0.155565709978604   0.097792483316752   0.017152606325211  -0.017632413884922
	//0.155565709977092  -0.821431705343740   0.260968359998222   0.215063954344560   0.003882506883519
	//0.097792483319198   0.260968359991318  -0.795890496175324   0.235746774334514   0.022352094439664
	//0.017152606324240   0.215063954350422   0.235746774329464  -0.723412687083199   0.067479872770964
	//-0.017632413884867   0.003882506882867   0.022352094440471   0.067479872770755  -0.427619755685488
	//

	SGMatrix<float64_t> L=inf->get_cholesky();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.582190787031602, rel_tolorance);
	EXPECT_NEAR(L(0,0),  -0.582190787031602,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.155565709978604, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.155565709978604,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.097792483316752, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.097792483316752,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.017152606325211, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.017152606325211,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.017632413884922, rel_tolorance);
	EXPECT_NEAR(L(0,4),  -0.017632413884922,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.155565709977092, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0.155565709977092,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.821431705343740, rel_tolorance);
	EXPECT_NEAR(L(1,1),  -0.821431705343740,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.260968359998222, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.260968359998222,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.215063954344560, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.215063954344560,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.003882506883519, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.003882506883519,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.097792483319198, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0.097792483319198,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.260968359991318, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0.260968359991318,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.795890496175324, rel_tolorance);
	EXPECT_NEAR(L(2,2),  -0.795890496175324,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.235746774334514, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.235746774334514,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.022352094439664, rel_tolorance);
	EXPECT_NEAR(L(2,4),  0.022352094439664,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.017152606324240, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0.017152606324240,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.215063954350422, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0.215063954350422,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.235746774329464, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0.235746774329464,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.723412687083199, rel_tolorance);
	EXPECT_NEAR(L(3,3),  -0.723412687083199,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.067479872770964, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.067479872770964,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.017632413884867, rel_tolorance);
	EXPECT_NEAR(L(4,0),  -0.017632413884867,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.003882506882867, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0.003882506882867,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.022352094440471, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0.022352094440471,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.067479872770755, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0.067479872770755,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.427619755685488, rel_tolorance);
	EXPECT_NEAR(L(4,4),  -0.427619755685488,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_cholesky_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);
	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior cholesky with result:
	// L =
	//-0.176739629317949   0.002314507852889   0.005090523727770   0.011082454381435   0.009003906401798
	//0.002314507852889  -0.169588536387473   0.003576487184405   0.020229751169513   0.018619008505207
	//0.005090523727770   0.003576487184405  -0.175680126714968   0.013003746250537   0.000889849322114
	//0.011082454381435   0.020229751169512   0.013003746250537  -0.178972749964575   0.016753869203624
	//0.009003906401798   0.018619008505207   0.000889849322114   0.016753869203624  -0.171929471862666
	// 


	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.176739629317949, rel_tolorance);
	EXPECT_NEAR(L(0,0),  -0.176739629317949,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.002314507852889, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.002314507852889,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.005090523727770, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.005090523727770,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.011082454381435, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.011082454381435,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.009003906401798, rel_tolorance);
	EXPECT_NEAR(L(0,4),  0.009003906401798,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.002314507852889, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0.002314507852889,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.169588536387473, rel_tolorance);
	EXPECT_NEAR(L(1,1),  -0.169588536387473,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.003576487184405, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.003576487184405,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020229751169513, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.020229751169513,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.018619008505207, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.018619008505207,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.005090523727770, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0.005090523727770,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.003576487184405, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0.003576487184405,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.175680126714968, rel_tolorance);
	EXPECT_NEAR(L(2,2),  -0.175680126714968,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.013003746250537, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.013003746250537,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000889849322114, rel_tolorance);
	EXPECT_NEAR(L(2,4),  0.000889849322114,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.011082454381435, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0.011082454381435,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020229751169512, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0.020229751169512,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.013003746250537, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0.013003746250537,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.178972749964575, rel_tolorance);
	EXPECT_NEAR(L(3,3),  -0.178972749964575,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.016753869203624, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.016753869203624,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.009003906401798, rel_tolorance);
	EXPECT_NEAR(L(4,0),  0.009003906401798,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.018619008505207, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0.018619008505207,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000889849322114, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0.000889849322114,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.016753869203624, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0.016753869203624,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.171929471862666, rel_tolorance);
	EXPECT_NEAR(L(4,4),  -0.171929471862666,  abs_tolorance);
	
	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_cholesky_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=10;
	// choose Gaussian kernel with sigma = 200 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior cholesky with result:
	// L =
	//-0.516395175487306   0.074256393102732   0.087938852628901   0.087869435051116   0.101286220204603
	//0.074256393102771  -0.451606829736600   0.051598292635512   0.086536297491332   0.084303187510036
	//0.087938852628916   0.051598292635512  -0.419635915905524   0.078693188687297   0.033820499045629
	//0.087869435051067   0.086536297491354   0.078693188687303  -0.458310607482338   0.068079128523596
	//0.101286220204592   0.084303187510047   0.033820499045638   0.068079128523585  -0.469848756925620
	// 

	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.516395175487306, rel_tolorance);
	EXPECT_NEAR(L(0,0),  -0.516395175487306,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.074256393102732, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.074256393102732,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.087938852628901, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.087938852628901,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.087869435051116, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.087869435051116,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.101286220204603, rel_tolorance);
	EXPECT_NEAR(L(0,4),  0.101286220204603,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.074256393102771, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0.074256393102771,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.451606829736600, rel_tolorance);
	EXPECT_NEAR(L(1,1),  -0.451606829736600,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.051598292635512, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.051598292635512,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086536297491332, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.086536297491332,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.084303187510036, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.084303187510036,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.087938852628916, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0.087938852628916,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.051598292635512, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0.051598292635512,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.419635915905524, rel_tolorance);
	EXPECT_NEAR(L(2,2),  -0.419635915905524,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.078693188687297, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.078693188687297,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.033820499045629, rel_tolorance);
	EXPECT_NEAR(L(2,4),  0.033820499045629,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.087869435051067, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0.087869435051067,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086536297491354, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0.086536297491354,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.078693188687303, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0.078693188687303,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.458310607482338, rel_tolorance);
	EXPECT_NEAR(L(3,3),  -0.458310607482338,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.068079128523596, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.068079128523596,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.101286220204592, rel_tolorance);
	EXPECT_NEAR(L(4,0),  0.101286220204592,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.084303187510047, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0.084303187510047,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.033820499045638, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0.033820499045638,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.068079128523585, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0.068079128523585,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.469848756925620, rel_tolorance);
	EXPECT_NEAR(L(4,4),  -0.469848756925620,  abs_tolorance);
	
	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_mean_t_likelihood)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTVGLikelihood* likelihood=new CStudentsTVGLikelihood(1, 3);

	// specify GP regression with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.489964784217491
	//0.847319427032666
	//0.889154609732680
	//0.946682723777487
	//0.814125173299711
	// 
	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
	abs_tolorance = CMath::get_abs_tolorance(0.489964784217491, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  0.489964784217491,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.847319427032666, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  0.847319427032666,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.889154609732680, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.889154609732680,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.946682723777487, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  0.946682723777487,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.814125173299711, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  0.814125173299711,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_covariance_t_likelihood)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTVGLikelihood* likelihood=new CStudentsTVGLikelihood(1, 3);

	// specify GP regression with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.414374171867591   0.143666984642054   0.094429894160241   0.017776219176013  -0.025310583164083
	//0.143666984642054   0.231294380280811   0.224883777583170   0.197483307920773   0.004953470086488
	//0.094429894160241   0.224883777583170   0.231567697418160   0.227352897220921   0.029816186766281
	//0.017776219176013   0.197483307920773   0.227352897220921   0.269271569986835   0.096502208542249
	//-0.025310583164083   0.004953470086488   0.029816186766281   0.096502208542249   0.560250749440729
	//
	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.414374171867591, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.414374171867591,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.143666984642054, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.143666984642054,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.094429894160241, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.094429894160241,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.017776219176013, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.017776219176013,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.025310583164083, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  -0.025310583164083,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.143666984642054, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.143666984642054,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.231294380280811, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.231294380280811,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.224883777583170, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.224883777583170,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.197483307920773, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.197483307920773,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.004953470086488, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.004953470086488,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.094429894160241, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.094429894160241,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.224883777583170, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.224883777583170,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.231567697418160, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.231567697418160,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.227352897220921, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.227352897220921,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.029816186766281, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  0.029816186766281,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.017776219176013, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.017776219176013,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.197483307920773, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.197483307920773,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.227352897220921, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.227352897220921,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.269271569986835, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.269271569986835,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.096502208542249, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.096502208542249,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.025310583164083, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  -0.025310583164083,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.004953470086488, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.004953470086488,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.029816186766281, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  0.029816186766281,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.096502208542249, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.096502208542249,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.560250749440729, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.560250749440729,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_mean_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.195075026543215
	//-0.752239731496555
	//0.254676687999610
	//-0.503016798226713
	//-0.666267997756778
	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.195075026543215, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  0.195075026543215,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.752239731496555, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  -0.752239731496555,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.254676687999610, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.254676687999610,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.503016798226713, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  -0.503016798226713,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.666267997756778, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  -0.666267997756778,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_covariance_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.788613876131317   0.055365607754537   0.113585694023662   0.250515273722783   0.211318428812988
	//0.055365607754537   0.705058940302301   0.086020320758721   0.492913804360438   0.471026706991197
	//0.113585694023662   0.086020320758721   0.795196807963229   0.295546542244499   0.020998208707717
	//0.250515273722783   0.492913804360438   0.295546542244499   0.677608376722589   0.400514223102444
	//0.211318428812988   0.471026706991197   0.020998208707717   0.400514223102444   0.714639742908927
	// 


	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.788613876131317, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.788613876131317,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.055365607754537, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.055365607754537,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.113585694023662, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.113585694023662,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.250515273722783, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.250515273722783,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.211318428812988, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  0.211318428812988,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.055365607754537, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.055365607754537,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.705058940302301, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.705058940302301,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086020320758721, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.086020320758721,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.492913804360438, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.492913804360438,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.471026706991197, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.471026706991197,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.113585694023662, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.113585694023662,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086020320758721, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.086020320758721,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.795196807963229, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.795196807963229,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.295546542244499, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.295546542244499,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020998208707717, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  0.020998208707717,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.250515273722783, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.250515273722783,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.492913804360438, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.492913804360438,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.295546542244499, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.295546542244499,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.677608376722589, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.677608376722589,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.400514223102444, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.400514223102444,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.211318428812988, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  0.211318428812988,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.471026706991197, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.471026706991197,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020998208707717, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  0.020998208707717,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.400514223102444, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.400514223102444,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.714639742908927, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.714639742908927,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_mean_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=10;
	// choose Gaussian kernel with sigma = 200 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);

	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.113297373330173
	//0.275513724397875
	//0.385527955754856
	//0.322231483097811
	//-0.062241963916682
	// 

	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.113297373330173, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  0.113297373330173,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.275513724397875, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  0.275513724397875,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.385527955754856, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.385527955754856,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.322231483097811, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  0.322231483097811,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.062241963916682, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  -0.062241963916682,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_posterior_covariance_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=10;
	// choose Gaussian kernel with sigma = 200 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.308346066129894   0.206868307317793   0.258036061496451   0.249092866479063   0.259977028483369
	//0.206868307317793   0.339676096729677   0.175180647367563   0.283840694594427   0.250369106608605
	//0.258036061496451   0.175180647367563   0.390420468722475   0.271865272760877   0.105793037165707
	//0.249092866479063   0.283840694594427   0.271865272760877   0.297244512625976   0.205738003168112
	//0.259977028483369   0.250369106608605   0.105793037165707   0.205738003168112   0.368531445290527
	// 

	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.308346066129894, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.308346066129894,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.206868307317793, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.206868307317793,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.258036061496451, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.258036061496451,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.249092866479063, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.249092866479063,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.259977028483369, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  0.259977028483369,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.206868307317793, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.206868307317793,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.339676096729677, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.339676096729677,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.175180647367563, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.175180647367563,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.283840694594427, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.283840694594427,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.250369106608605, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.250369106608605,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.258036061496451, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.258036061496451,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.175180647367563, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.175180647367563,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.390420468722475, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.390420468722475,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.271865272760877, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.271865272760877,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.105793037165707, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  0.105793037165707,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.249092866479063, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.249092866479063,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.283840694594427, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.283840694594427,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.271865272760877, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.271865272760877,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.297244512625976, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.297244512625976,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.205738003168112, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.205738003168112,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.259977028483369, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  0.259977028483369,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.250369106608605, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.250369106608605,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.105793037165707, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  0.105793037165707,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.205738003168112, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.205738003168112,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.368531445290527, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.368531445290527,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_negative_marginal_likelihood_t_likelihood)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTVGLikelihood* likelihood=new CStudentsTVGLikelihood(1, 3);

	// specify GP regression with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// nlZ =
	// 7.383353794839424
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(7.383353794839424, rel_tolorance);
	EXPECT_NEAR(nml, 7.383353794839424, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	//3.359093542091830
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(3.359093542091830, rel_tolorance);
	EXPECT_NEAR(nml, 3.359093542091830, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=10;
	// choose Gaussian kernel with sigma = 200 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	//3.900050836490685
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;


 	abs_tolorance = CMath::get_abs_tolorance(3.900050836490685, rel_tolorance);
	EXPECT_NEAR(nml, 3.900050836490685, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_marginal_likelihood_derivatives_t_likelihood)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5;

	SGMatrix<float64_t> feat_train(1, ntr);
	SGVector<float64_t> lab_train(ntr);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	float64_t ell=0.1;

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 0.25, df = 3
	CStudentsTVGLikelihood* lik=new CStudentsTVGLikelihood(0.25, 3);

	// specify GP regression with exact inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
		features_train,	mean, labels_train, lik);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");
	TParameter* df_param=lik->m_gradient_parameters->get_parameter("df");

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_df=(gradient->get_element(df_param))[0];
	float64_t dnlZ_sigma=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// lik =
	//-0.208254633811686
	//0.024939620577627
	// 
	// 
	// cov =
	//-0.805849415403716
	//-0.418521621175371
	// 

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.208254633811686, rel_tolorance);
	EXPECT_NEAR(dnlZ_df, -0.208254633811686, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.024939620577627, rel_tolorance);
	EXPECT_NEAR(dnlZ_sigma, 0.024939620577627, abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.805849415403716, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.805849415403716, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.418521621175371, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.418521621175371, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");

	float64_t dnlZ_ell=4.0*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	// 0.275308238001720
	//-0.138232607204219
	
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.275308238001720, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, 0.275308238001720, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.138232607204219, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.138232607204219, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLCholeskyInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=10;
	// choose Gaussian kernel with sigma = 200 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLCholeskyInferenceMethod* inf=new CKLCholeskyInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/bb400ebded2dbe90c58d
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	//0.299593140038451
	//0.401789648589235
	// 
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.299593140038451, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, 0.299593140038451, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.401789648589235, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, 0.401789648589235, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}


#endif /* HAVE_EIGEN3 */
