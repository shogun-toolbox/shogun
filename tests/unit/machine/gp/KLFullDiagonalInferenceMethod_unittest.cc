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
#include <shogun/machine/gp/KLFullDiagonalInferenceMethod.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/machine/gp/ProbitVGLikelihood.h>
#include <shogun/machine/gp/StudentsTVGLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

TEST(KLFullDiagonalInferenceMethod,get_cholesky_t_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// L =
	//
	//1.407423685644243   0.515497460024622   0.443065894253950   0.319410344939164   0.016180892181750
	//                0   1.355918485894513   0.619918738844402   0.588106670839222   0.115276398753254
	//                0                   0   1.214360350469977   0.394486954116069   0.104931348884614
	//                0                   0                   0   1.175209056259849   0.157509505562212
	//                0                   0                   0                   0   1.288734252271706
	//
	SGMatrix<float64_t> L=inf->get_cholesky();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
	abs_tolorance = CMath::get_abs_tolorance(1.407423685644243, rel_tolorance);
	EXPECT_NEAR(L(0,0),  1.407423685644243,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.515497460024622, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.515497460024622,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.443065894253950, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.443065894253950,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.319410344939164, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.319410344939164,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.016180892181750, rel_tolorance);
	EXPECT_NEAR(L(0,4),  0.016180892181750,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.355918485894513, rel_tolorance);
	EXPECT_NEAR(L(1,1),  1.355918485894513,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.619918738844402, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.619918738844402,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.588106670839222, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.588106670839222,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.115276398753254, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.115276398753254,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.214360350469977, rel_tolorance);
	EXPECT_NEAR(L(2,2),  1.214360350469977,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.394486954116069, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.394486954116069,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.104931348884614, rel_tolorance);
	EXPECT_NEAR(L(2,4),  0.104931348884614,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.175209056259849, rel_tolorance);
	EXPECT_NEAR(L(3,3),  1.175209056259849,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.157509505562212, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.157509505562212,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.288734252271706, rel_tolorance);
	EXPECT_NEAR(L(4,4),  1.288734252271706,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_cholesky_logit_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior cholesky with result:
	// L =
	//1.101033636127231   0.031935435514485   0.038838470404416   0.080505388829644   0.067456404131013
	//                0   1.093577781758797   0.034349943478487   0.142357943078213   0.133350176799401
	//                0                   0   1.099289233197989   0.080619635381560   0.013257570410259
	//                0                   0                   0   1.084046011513489   0.103671309031576
	//                0                   0                   0                   0   1.080493372046284


	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(1.101033636127231, rel_tolorance);
	EXPECT_NEAR(L(0,0),  1.101033636127231,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.031935435514485, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.031935435514485,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.038838470404416, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.038838470404416,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.080505388829644, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.080505388829644,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.067456404131013, rel_tolorance);
	EXPECT_NEAR(L(0,4),  0.067456404131013,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.093577781758797, rel_tolorance);
	EXPECT_NEAR(L(1,1),  1.093577781758797,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.034349943478487, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.034349943478487,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.142357943078213, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.142357943078213,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.133350176799401, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.133350176799401,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.099289233197989, rel_tolorance);
	EXPECT_NEAR(L(2,2),  1.099289233197989,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.080619635381560, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.080619635381560,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.013257570410259, rel_tolorance);
	EXPECT_NEAR(L(2,4),  0.013257570410259,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.084046011513489, rel_tolorance);
	EXPECT_NEAR(L(3,3),  1.084046011513489,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.103671309031576, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.103671309031576,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.080493372046284, rel_tolorance);
	EXPECT_NEAR(L(4,4),  1.080493372046284,  abs_tolorance);
	
	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_cholesky_probit_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior cholesky with result:
	// L =
	//1.216420672271186   0.000000401780554   0.000004054670173   0.000949334832622   0.000022647889665
	//                0   1.215717061852584   0.000000000003300   0.006432557084865   0.000001816526618
	//                0                   0   1.216327485101524   0.000048851822686  -0.000000000075498
	//                0                   0                   0   1.215786494210340   0.000000009487969
	//                0                   0                   0                   0   1.216329768018010


	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(1.216420672271186, rel_tolorance);
	EXPECT_NEAR(L(0,0),  1.216420672271186,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000000401780554, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.000000401780554,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000004054670173, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.000004054670173,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000949334832622, rel_tolorance);
	EXPECT_NEAR(L(0,3),  0.000949334832622,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000022647889665, rel_tolorance);
	EXPECT_NEAR(L(0,4),  0.000022647889665,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.215717061852584, rel_tolorance);
	EXPECT_NEAR(L(1,1),  1.215717061852584,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000000000003300, rel_tolorance);
	EXPECT_NEAR(L(1,2),  0.000000000003300,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.006432557084865, rel_tolorance);
	EXPECT_NEAR(L(1,3),  0.006432557084865,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000001816526618, rel_tolorance);
	EXPECT_NEAR(L(1,4),  0.000001816526618,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(2,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.216327485101524, rel_tolorance);
	EXPECT_NEAR(L(2,2),  1.216327485101524,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000048851822686, rel_tolorance);
	EXPECT_NEAR(L(2,3),  0.000048851822686,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000000075498, rel_tolorance);
	EXPECT_NEAR(L(2,4),  -0.000000000075498,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(3,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.215786494210340, rel_tolorance);
	EXPECT_NEAR(L(3,3),  1.215786494210340,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000000009487969, rel_tolorance);
	EXPECT_NEAR(L(3,4),  0.000000009487969,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,0),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,1),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,2),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0, rel_tolorance);
	EXPECT_NEAR(L(4,3),  0,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(1.216329768018010, rel_tolorance);
	EXPECT_NEAR(L(4,4),  1.216329768018010,  abs_tolorance);
	
	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_mean_t_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.489965084229748
	//0.847314396373669
	//0.889161956923399
	//0.946680542914168
	//0.814125447090175
	//
	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
	abs_tolorance = CMath::get_abs_tolorance(0.489965084229748, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  0.489965084229748,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.847314396373669, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  0.847314396373669,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.889161956923399, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.889161956923399,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.946680542914168, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  0.946680542914168,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.814125447090175, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  0.814125447090175,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_covariance_t_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero covariance function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* covariance=new CZeroMean();

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTVGLikelihood* likelihood=new CStudentsTVGLikelihood(1, 3);

	// specify GP regression with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	covariance, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.414373949946085   0.143666705039313   0.094429636121475   0.017776032079365  -0.025310137089717
	//0.143666705039313   0.231294061589554   0.224883475133982   0.197483047433658   0.004953399924042
	//0.094429636121475   0.224883475133982   0.231567407777062   0.227352640803712   0.029816060215331
	//0.017776032079365   0.197483047433658   0.227352640803712   0.269271328474079   0.096502018273666
	//-0.025310137089717   0.004953399924042   0.029816060215331   0.096502018273666   0.560250726200816
	//
	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
	abs_tolorance = CMath::get_abs_tolorance(0.414373949946085, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.414373949946085,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.143666705039313, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.143666705039313,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.094429636121475, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.094429636121475,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.017776032079365, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.017776032079365,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.025310137089717, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  -0.025310137089717,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.143666705039313, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.143666705039313,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.231294061589554, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.231294061589554,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.224883475133982, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.224883475133982,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.197483047433658, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.197483047433658,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.004953399924042, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.004953399924042,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.094429636121475, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.094429636121475,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.224883475133982, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.224883475133982,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.231567407777062, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.231567407777062,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.227352640803712, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.227352640803712,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.029816060215331, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  0.029816060215331,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.017776032079365, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.017776032079365,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.197483047433658, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.197483047433658,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.227352640803712, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.227352640803712,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.269271328474079, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.269271328474079,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.096502018273666, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.096502018273666,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.025310137089717, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  -0.025310137089717,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.004953399924042, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.004953399924042,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.029816060215331, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  0.029816060215331,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.096502018273666, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.096502018273666,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.560250726200816, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.560250726200816,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_mean_logit_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.195075001254701
	//-0.752239725258407
	//0.254676683453462
	//-0.503016798305133
	//-0.666267998053423
	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.195075001254701, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  0.195075001254701,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.752239725258407, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  -0.752239725258407,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.254676683453462, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.254676683453462,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.503016798305133, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  -0.503016798305133,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.666267998053423, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  -0.666267998053423,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_covariance_logit_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.788613922080216   0.055365603952762   0.113585701478720   0.250515288102739   0.211318414874201
	//0.055365603952762   0.705059022635855   0.086020351983001   0.492913876934693   0.471026718609916
	//0.113585701478720   0.086020351983001   0.795196757855660   0.295546558157168   0.020998235505635
	//0.250515288102739   0.492913876934693   0.295546558157168   0.677608447921106   0.400514239283206
	//0.211318414874201   0.471026718609916   0.020998235505635   0.400514239283206   0.714639675393856
	// 

	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.788613922080216, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.788613922080216,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.055365603952762, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.055365603952762,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.113585701478720, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.113585701478720,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.250515288102739, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.250515288102739,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.211318414874201, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  0.211318414874201,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.055365603952762, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.055365603952762,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.705059022635855, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.705059022635855,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086020351983001, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.086020351983001,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.492913876934693, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.492913876934693,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.471026718609916, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.471026718609916,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.113585701478720, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.113585701478720,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.086020351983001, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.086020351983001,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.795196757855660, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.795196757855660,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.295546558157168, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.295546558157168,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020998235505635, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  0.020998235505635,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.250515288102739, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.250515288102739,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.492913876934693, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.492913876934693,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.295546558157168, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.295546558157168,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.677608447921106, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.677608447921106,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.400514239283206, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.400514239283206,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.211318414874201, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  0.211318414874201,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.471026718609916, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.471026718609916,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.020998235505635, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  0.020998235505635,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.400514239283206, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.400514239283206,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.714639675393856, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.714639675393856,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_mean_probit_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//-0.562633450485667
	//0.569793845834907
	//0.563581584765779
	//0.568910892345396
	//-0.563558402061117
	// 

	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.562633450485667, rel_tolorance);
	EXPECT_NEAR(posterior_mean[0],  -0.562633450485667,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.569793845834907, rel_tolorance);
	EXPECT_NEAR(posterior_mean[1],  0.569793845834907,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.563581584765779, rel_tolorance);
	EXPECT_NEAR(posterior_mean[2],  0.563581584765779,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.568910892345396, rel_tolorance);
	EXPECT_NEAR(posterior_mean[3],  0.568910892345396,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.563558402061117, rel_tolorance);
	EXPECT_NEAR(posterior_mean[4],  -0.563558402061117,  abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_posterior_covariance_probit_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.675821281453662  -0.000005367710037   0.000004653883433   0.001102425357803   0.000026241600793
	//-0.000005367710037   0.676565020447684  -0.000000300323282   0.007487585977222   0.000002109695855
	//0.000004653883433  -0.000000300323282   0.675925696916253   0.000056747602632  -0.000000000087947
	//0.001102425357803   0.007487585977222   0.000056747602632   0.676467549296978   0.000000011016475
	//0.000026241600793   0.000002109695855  -0.000000000087947   0.000000011016475   0.675923161214598

	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.675821281453662, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.675821281453662,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000005367710037, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,1),  -0.000005367710037,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000004653883433, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.000004653883433,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.001102425357803, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.001102425357803,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000026241600793, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(0,4),  0.000026241600793,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.000005367710037, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,0),  -0.000005367710037,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.676565020447684, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.676565020447684,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000300323282, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,2),  -0.000000300323282,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.007487585977222, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.007487585977222,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000002109695855, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.000002109695855,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.000004653883433, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.000004653883433,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000300323282, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,1),  -0.000000300323282,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.675925696916253, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.675925696916253,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000056747602632, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.000056747602632,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000000087947, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(2,4),  -0.000000000087947,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.001102425357803, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.001102425357803,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.007487585977222, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.007487585977222,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000056747602632, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.000056747602632,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.676467549296978, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.676467549296978,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000000011016475, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.000000011016475,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(0.000026241600793, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,0),  0.000026241600793,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000002109695855, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.000002109695855,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000000087947, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,2),  -0.000000000087947,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.000000011016475, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.000000011016475,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.675923161214598, rel_tolorance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.675923161214598,  abs_tolorance);
	
	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_negative_marginal_likelihood_t_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// nlZ =
	// 7.38335326307118311462
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(7.38335326307118311462, rel_tolorance);
	EXPECT_NEAR(nml, 7.38335326307118311462, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	// 3.359093542091840
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(3.359093542091840, rel_tolorance);
	EXPECT_NEAR(nml, 3.359093542091840, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	//3.468786292404183
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(3.468786292404183, rel_tolorance);
	EXPECT_NEAR(nml, 3.468786292404183, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_marginal_likelihood_derivatives_t_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
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
	float64_t dnlZ_sf2=1.0*(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// lik =
	//-0.208254635605496
	//0.024939622917056
	// 
	// cov =
	//-0.80584941351205596760    
	//-0.41852168768504860452

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.208254635605496, rel_tolorance);
	EXPECT_NEAR(dnlZ_df, -0.208254635605496, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.024939622917056, rel_tolorance);
	EXPECT_NEAR(dnlZ_sigma, 0.024939622917056, abs_tolorance);

	abs_tolorance = CMath::get_abs_tolorance(-0.80584941351205596760, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.80584941351205596760, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.41852168768504860452, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.41852168768504860452, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
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
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
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

	float64_t dnlZ_ell=4*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=1.0*(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	//0.275308215764774
	//-0.138232606081787

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(0.275308215764774, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, 0.275308215764774, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.138232606081787, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.138232606081787, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(KLFullDiagonalInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
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

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitVGLikelihood* likelihood=new CProbitVGLikelihood();

	// specify GP classification with KL inference
	CKLFullDiagonalInferenceMethod* inf=new CKLFullDiagonalInferenceMethod(kernel,
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

	float64_t dnlZ_ell=4*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=1.0*(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/b64a015491833562d11a
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	//-0.034304800769586
	//0.028091203761949
	
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.034304800769586, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.034304800769586, abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.028091203761949, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, 0.028091203761949, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}


#endif /* HAVE_EIGEN3 */
