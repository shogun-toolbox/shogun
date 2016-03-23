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
 */
#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/KLApproxDiagonalInferenceMethod.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/machine/gp/ProbitVGLikelihood.h>
#include <shogun/machine/gp/StudentsTVGLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

TEST(KLApproxDiagonalInferenceMethod,get_cholesky_t_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// L =
	//
	//29.63405683123 -612.02565401263 856.26722752838 -280.16777275207 10.21240287952
	//-612.02565401263 11738.61130584184 -16309.89349742437 5283.49601888462 -193.58285625197
	//856.26722752838 -16309.89349742438 22628.70482752738 -7311.61034280088 267.29360984377
	//-280.16777275206 5283.49601888462 -7311.61034280087 2350.18015777553 -85.27403983871
	//10.21240287952 -193.58285625197 267.29360984377 -85.27403983871 2.37644470776

	SGMatrix<float64_t> L=inf->get_cholesky();

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(29.63405683123, rel_tolerance);
	EXPECT_NEAR(L(0,0),  29.63405683123,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-612.02565401263, rel_tolerance);
	EXPECT_NEAR(L(0,1),  -612.02565401263,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(856.26722752838, rel_tolerance);
	EXPECT_NEAR(L(0,2),  856.26722752838,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-280.16777275207, rel_tolerance);
	EXPECT_NEAR(L(0,3),  -280.16777275207,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(10.21240287952, rel_tolerance);
	EXPECT_NEAR(L(0,4),  10.21240287952,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-612.02565401263, rel_tolerance);
	EXPECT_NEAR(L(1,0),  -612.02565401263,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(11738.61130584184, rel_tolerance);
	EXPECT_NEAR(L(1,1),  11738.61130584184,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-16309.89349742437, rel_tolerance);
	EXPECT_NEAR(L(1,2),  -16309.89349742437,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(5283.49601888462, rel_tolerance);
	EXPECT_NEAR(L(1,3),  5283.49601888462,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-193.58285625197, rel_tolerance);
	EXPECT_NEAR(L(1,4),  -193.58285625197,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(856.26722752838, rel_tolerance);
	EXPECT_NEAR(L(2,0),  856.26722752838,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-16309.89349742438, rel_tolerance);
	EXPECT_NEAR(L(2,1),  -16309.89349742438,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(22628.70482752738, rel_tolerance);
	EXPECT_NEAR(L(2,2),  22628.70482752738,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-7311.61034280088, rel_tolerance);
	EXPECT_NEAR(L(2,3),  -7311.61034280088,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(267.29360984377, rel_tolerance);
	EXPECT_NEAR(L(2,4),  267.29360984377,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-280.16777275206, rel_tolerance);
	EXPECT_NEAR(L(3,0),  -280.16777275206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(5283.49601888462, rel_tolerance);
	EXPECT_NEAR(L(3,1),  5283.49601888462,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-7311.61034280087, rel_tolerance);
	EXPECT_NEAR(L(3,2),  -7311.61034280087,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2350.18015777553, rel_tolerance);
	EXPECT_NEAR(L(3,3),  2350.18015777553,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-85.27403983871, rel_tolerance);
	EXPECT_NEAR(L(3,4),  -85.27403983871,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(10.21240287952, rel_tolerance);
	EXPECT_NEAR(L(4,0),  10.21240287952,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-193.58285625197, rel_tolerance);
	EXPECT_NEAR(L(4,1),  -193.58285625197,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(267.29360984377, rel_tolerance);
	EXPECT_NEAR(L(4,2),  267.29360984377,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-85.27403983871, rel_tolerance);
	EXPECT_NEAR(L(4,3),  -85.27403983871,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.37644470776, rel_tolerance);
	EXPECT_NEAR(L(4,4),  2.37644470776,  abs_tolerance);


	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_cholesky_logit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior cholesky with result:
	// L =
	//0.305566275391344   1.588768886781588   0.319850350212507  -1.286405366165323  -0.648640619917849
	//1.588768886781589   2.761020834442978   0.924176676583413  -2.976289429766340  -1.256843239461661
	//0.319850350212507   0.924176676583413   0.259022764368172  -1.425380890787893   0.257140995693305
	//-1.286405366165324  -2.976289429766339  -1.425380890787893   3.099155969153228   0.557830504105774
	//-0.648640619917849  -1.256843239461661   0.257140995693305   0.557830504105774   0.691714117008702
	//



	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.305566275391344, rel_tolerance);
	EXPECT_NEAR(L(0,0),  0.305566275391344,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.588768886781588, rel_tolerance);
	EXPECT_NEAR(L(0,1),  1.588768886781588,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.319850350212507, rel_tolerance);
	EXPECT_NEAR(L(0,2),  0.319850350212507,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.286405366165323, rel_tolerance);
	EXPECT_NEAR(L(0,3),  -1.286405366165323,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.648640619917849, rel_tolerance);
	EXPECT_NEAR(L(0,4),  -0.648640619917849,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(1.588768886781589, rel_tolerance);
	EXPECT_NEAR(L(1,0),  1.588768886781589,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.761020834442978, rel_tolerance);
	EXPECT_NEAR(L(1,1),  2.761020834442978,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.924176676583413, rel_tolerance);
	EXPECT_NEAR(L(1,2),  0.924176676583413,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2.976289429766340, rel_tolerance);
	EXPECT_NEAR(L(1,3),  -2.976289429766340,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.256843239461661, rel_tolerance);
	EXPECT_NEAR(L(1,4),  -1.256843239461661,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.319850350212507, rel_tolerance);
	EXPECT_NEAR(L(2,0),  0.319850350212507,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.924176676583413, rel_tolerance);
	EXPECT_NEAR(L(2,1),  0.924176676583413,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.259022764368172, rel_tolerance);
	EXPECT_NEAR(L(2,2),  0.259022764368172,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.425380890787893, rel_tolerance);
	EXPECT_NEAR(L(2,3),  -1.425380890787893,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.257140995693305, rel_tolerance);
	EXPECT_NEAR(L(2,4),  0.257140995693305,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-1.286405366165324, rel_tolerance);
	EXPECT_NEAR(L(3,0),  -1.286405366165324,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2.976289429766339, rel_tolerance);
	EXPECT_NEAR(L(3,1),  -2.976289429766339,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.425380890787893, rel_tolerance);
	EXPECT_NEAR(L(3,2),  -1.425380890787893,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(3.099155969153228, rel_tolerance);
	EXPECT_NEAR(L(3,3),  3.099155969153228,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.557830504105774, rel_tolerance);
	EXPECT_NEAR(L(3,4),  0.557830504105774,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-0.648640619917849, rel_tolerance);
	EXPECT_NEAR(L(4,0),  -0.648640619917849,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.256843239461661, rel_tolerance);
	EXPECT_NEAR(L(4,1),  -1.256843239461661,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.257140995693305, rel_tolerance);
	EXPECT_NEAR(L(4,2),  0.257140995693305,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.557830504105774, rel_tolerance);
	EXPECT_NEAR(L(4,3),  0.557830504105774,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.691714117008702, rel_tolerance);
	EXPECT_NEAR(L(4,4),  0.691714117008702,  abs_tolerance);


	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_cholesky_probit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior cholesky with result:
	// L =
	//92.55758851904 132.10723436942 23.49497816231 -182.27780561275 -62.82862008990
	//132.10723436942 202.42407260746 55.41981753776 -300.39377119259 -81.76913567091
	//23.49497816231 55.41981753776 22.07959299455 -84.71105676976 -12.95847578171
	//-182.27780561275 -300.39377119259 -84.71105676976 438.05343106817 116.71505545158
	//-62.82862008990 -81.76913567091 -12.95847578171 116.71505545158 38.71428242973

	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(92.55758851904, rel_tolerance);
	EXPECT_NEAR(L(0,0),  92.55758851904,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(132.10723436942, rel_tolerance);
	EXPECT_NEAR(L(0,1),  132.10723436942,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(23.49497816231, rel_tolerance);
	EXPECT_NEAR(L(0,2),  23.49497816231,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-182.27780561275, rel_tolerance);
	EXPECT_NEAR(L(0,3),  -182.27780561275,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-62.82862008990, rel_tolerance);
	EXPECT_NEAR(L(0,4),  -62.82862008990,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(132.10723436942, rel_tolerance);
	EXPECT_NEAR(L(1,0),  132.10723436942,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(202.42407260746, rel_tolerance);
	EXPECT_NEAR(L(1,1),  202.42407260746,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(55.41981753776, rel_tolerance);
	EXPECT_NEAR(L(1,2),  55.41981753776,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-300.39377119259, rel_tolerance);
	EXPECT_NEAR(L(1,3),  -300.39377119259,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-81.76913567091, rel_tolerance);
	EXPECT_NEAR(L(1,4),  -81.76913567091,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(23.49497816231, rel_tolerance);
	EXPECT_NEAR(L(2,0),  23.49497816231,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(55.41981753776, rel_tolerance);
	EXPECT_NEAR(L(2,1),  55.41981753776,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(22.07959299455, rel_tolerance);
	EXPECT_NEAR(L(2,2),  22.07959299455,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-84.71105676976, rel_tolerance);
	EXPECT_NEAR(L(2,3),  -84.71105676976,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-12.95847578171, rel_tolerance);
	EXPECT_NEAR(L(2,4),  -12.95847578171,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-182.27780561275, rel_tolerance);
	EXPECT_NEAR(L(3,0),  -182.27780561275,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-300.39377119259, rel_tolerance);
	EXPECT_NEAR(L(3,1),  -300.39377119259,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-84.71105676976, rel_tolerance);
	EXPECT_NEAR(L(3,2),  -84.71105676976,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(438.05343106817, rel_tolerance);
	EXPECT_NEAR(L(3,3),  438.05343106817,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(116.71505545158, rel_tolerance);
	EXPECT_NEAR(L(3,4),  116.71505545158,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-62.82862008990, rel_tolerance);
	EXPECT_NEAR(L(4,0),  -62.82862008990,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-81.76913567091, rel_tolerance);
	EXPECT_NEAR(L(4,1),  -81.76913567091,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-12.95847578171, rel_tolerance);
	EXPECT_NEAR(L(4,2),  -12.95847578171,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(116.71505545158, rel_tolerance);
	EXPECT_NEAR(L(4,3),  116.71505545158,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(38.71428242973, rel_tolerance);
	EXPECT_NEAR(L(4,4),  38.71428242973,  abs_tolerance);


	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_mean_t_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.504263183608509
	//0.877108239105726
	//0.922083624942563
	//0.985347268991102
	//0.860823087684119

	//
	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;
	abs_tolerance = CMath::get_abs_tolerance(0.504263183608509, rel_tolerance);
	EXPECT_NEAR(posterior_mean[0],  0.504263183608509,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.877108239105726, rel_tolerance);
	EXPECT_NEAR(posterior_mean[1],  0.877108239105726,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.922083624942563, rel_tolerance);
	EXPECT_NEAR(posterior_mean[2],  0.922083624942563,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.985347268991102, rel_tolerance);
	EXPECT_NEAR(posterior_mean[3],  0.985347268991102,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.860823087684119, rel_tolerance);
	EXPECT_NEAR(posterior_mean[4],  0.860823087684119,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_covariance_t_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.066339535327499                   0                   0                   0                   0
	//0   0.000246981976988                   0                   0                   0
	//0                   0   0.000128328494271                   0                   0
	//0                   0                   0   0.001199236426520                   0
	//0                   0                   0                   0   0.318754271084796

	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.066339535327499, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.066339535327499,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000246981976988, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.000246981976988,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000128328494271, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.000128328494271,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001199236426520, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.001199236426520,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.318754271084796, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.318754271084796,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_mean_logit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.198743176851626
	//-0.734494838264326
	//0.257097960876591
	//-0.487597222711448
	//-0.650341037910017

	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.198743176851626, rel_tolerance);
	EXPECT_NEAR(posterior_mean[0],  0.198743176851626,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.734494838264326, rel_tolerance);
	EXPECT_NEAR(posterior_mean[1],  -0.734494838264326,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.257097960876591, rel_tolerance);
	EXPECT_NEAR(posterior_mean[2],  0.257097960876591,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.487597222711448, rel_tolerance);
	EXPECT_NEAR(posterior_mean[3],  -0.487597222711448,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.650341037910017, rel_tolerance);
	EXPECT_NEAR(posterior_mean[4],  -0.650341037910017,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_covariance_logit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.603527438532628                   0                   0                   0                   0
	//0   0.238101224497216                   0                   0                   0
	//0                   0   0.595875427812964                   0                   0
	//0                   0                   0   0.215037761065975                   0
	//0                   0                   0                   0   0.347692051574632


	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.603527438532628, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.603527438532628,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.238101224497216, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.238101224497216,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.595875427812964, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.595875427812964,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.215037761065975, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.215037761065975,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.347692051574632, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.347692051574632,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_mean_probit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.103563174064572
	//0.259496134122113
	//0.363204885226123
	//0.303312594979317
	//-0.062882473885823

	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.103563174064572, rel_tolerance);
	EXPECT_NEAR(posterior_mean[0],  0.103563174064572,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.259496134122113, rel_tolerance);
	EXPECT_NEAR(posterior_mean[1],  0.259496134122113,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.363204885226123, rel_tolerance);
	EXPECT_NEAR(posterior_mean[2],  0.363204885226123,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.303312594979317, rel_tolerance);
	EXPECT_NEAR(posterior_mean[3],  0.303312594979317,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.062882473885823, rel_tolerance);
	EXPECT_NEAR(posterior_mean[4],  -0.062882473885823,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_posterior_covariance_probit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.017367535014875                   0                   0                   0                   0
	//0   0.010154165926107                   0                   0                   0
	//0                   0   0.036287922704977                   0                   0
	//0                   0                   0   0.004582936288132                   0
	//0                   0                   0                   0   0.036595981007138


	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.017367535014875, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.017367535014875,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.010154165926107, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.010154165926107,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.036287922704977, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.036287922704977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.004582936288132, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.004582936288132,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,4),  0,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.036595981007138, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.036595981007138,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_negative_marginal_likelihood_t_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// nlZ =
	//13.703509228407690
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(13.703509228407690, rel_tolerance);
	EXPECT_NEAR(nml, 13.703509228407690, abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	//4.160543707929570
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(4.160543707929570, rel_tolerance);
	EXPECT_NEAR(nml, 4.160543707929570, abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	// 7.744342191099269
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(7.744342191099269, rel_tolerance);
	EXPECT_NEAR(nml, 7.744342191099269, abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_marginal_likelihood_derivatives_t_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, lik);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("log_sigma");
	TParameter* df_param=lik->m_gradient_parameters->get_parameter("log_df");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_df=(gradient->get_element(df_param))[0];
	float64_t dnlZ_sigma=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// lik =
	//-0.208510139631534
	//0.027396140581594
	// cov =
	//-0.802268825425508
	//-0.421067396716859

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(-0.208510139631534, rel_tolerance);
	EXPECT_NEAR(dnlZ_df, -0.208510139631534, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance( 0.027396140581594, rel_tolerance);
	EXPECT_NEAR(dnlZ_sigma, 0.027396140581594, abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-0.802268825425508, rel_tolerance);
	EXPECT_NEAR(dnlZ_ell, -0.802268825425508, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.421067396716859, rel_tolerance);
	EXPECT_NEAR(dnlZ_sf2, -0.421067396716859, abs_tolerance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	//2.285921615561960
	//-0.432276087896124

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(2.285921615561960, rel_tolerance);
	EXPECT_NEAR(dnlZ_ell, 2.285921615561960, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.432276087896124, rel_tolerance);
	EXPECT_NEAR(dnlZ_sf2, -0.432276087896124, abs_tolerance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLApproxDiagonalInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
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
	CKLApproxDiagonalInferenceMethod* inf=new CKLApproxDiagonalInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/d8acb388d03c6976728e
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	//4.180258730739274
	//-0.462089650290405

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(4.180258730739274, rel_tolerance);
	EXPECT_NEAR(dnlZ_ell, 4.180258730739274, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.462089650290405, rel_tolerance);
	EXPECT_NEAR(dnlZ_sf2, -0.462089650290405, abs_tolerance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

