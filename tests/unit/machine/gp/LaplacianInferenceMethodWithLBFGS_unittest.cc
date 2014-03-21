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
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * 
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/LaplacianInferenceMethodWithLBFGS.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

float64_t get_abs_tolorance_lbfgs(float64_t true_value, float64_t rel_tolorance)
{
	rel_tolorance = CMath::abs(rel_tolorance);
	return true_value==0.0 ? rel_tolorance : CMath::abs(true_value * rel_tolorance);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_cholesky_probit_likelihood)
{
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);


	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*L =
		1.229795134715245   0.000000424572149   0.000004284193391   0.001003171160332   0.000023929911208
		0.000000000000000   1.229105947500809   0.000000000003423   0.006798109390543   0.000001919569807
		0.000000000000000   0.000000000000000   1.229704014299049   0.000051622815564  -0.000000000083370
		0.000000000000000   0.000000000000000   0.000000000000000   1.229171826524023   0.000000008722897
		0.000000000000000   0.000000000000000   0.000000000000000   0.000000000000000   1.229706246172444
	*/
	SGMatrix<float64_t> L=inf->get_cholesky();

	abs_tolorance = get_abs_tolorance_lbfgs(1.229795134715245, rel_tolorance);
	EXPECT_NEAR(L(0,0), 1.229795134715245, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000000424572149, rel_tolorance);
	EXPECT_NEAR(L(0,1), 0.000000424572149, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000004284193391, rel_tolorance);
	EXPECT_NEAR(L(0,2), 0.000004284193391, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.001003171160332, rel_tolorance);
	EXPECT_NEAR(L(0,3), 0.001003171160332, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000023929911208, rel_tolorance);
	EXPECT_NEAR(L(0,4), 0.000023929911208, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.229105947500809, rel_tolorance);
	EXPECT_NEAR(L(1,1), 1.229105947500809, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000000000003423, rel_tolorance);
	EXPECT_NEAR(L(1,2), 0.000000000003423, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.006798109390543, rel_tolorance);
	EXPECT_NEAR(L(1,3), 0.006798109390543, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000001919569807, rel_tolorance);
	EXPECT_NEAR(L(1,4), 0.000001919569807, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.229704014299049, rel_tolorance);
	EXPECT_NEAR(L(2,2), 1.229704014299049, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000051622815564, rel_tolorance);
	EXPECT_NEAR(L(2,3), 0.000051622815564, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.000000000083370, rel_tolorance);
	EXPECT_NEAR(L(2,4), -0.000000000083370, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.229171826524023, rel_tolorance);
	EXPECT_NEAR(L(3,3), 1.229171826524023, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000000008722897, rel_tolorance);
	EXPECT_NEAR(L(3,4), 0.000000008722897, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.229706246172444, rel_tolorance);
	EXPECT_NEAR(L(4,4), 1.229706246172444, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_alpha_probit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*alpha = 
		-0.506457945471096
		0.503267616409653
		0.506035061915211
		0.503660487331861
		-0.506045417007059
	*/

	SGVector<float64_t> alpha=inf->get_alpha();

	abs_tolorance = get_abs_tolorance_lbfgs(-0.506457945471096, rel_tolorance);
	EXPECT_NEAR(alpha[0], -0.506457945471096, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.503267616409653, rel_tolorance);
	EXPECT_NEAR(alpha[1], 0.503267616409653, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.506035061915211, rel_tolorance);
	EXPECT_NEAR(alpha[2], 0.506035061915211, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.503660487331861, rel_tolorance);
	EXPECT_NEAR(alpha[3], 0.503660487331861, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.506045417007059, rel_tolorance);
	EXPECT_NEAR(alpha[4], -0.506045417007059, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_negative_marginal_likelihood_probit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*nlZ =
		3.499023867961728
	*/
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	abs_tolorance = get_abs_tolorance_lbfgs(3.499023867961728, rel_tolorance);
	EXPECT_NEAR(nml, 3.499023867961728, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_marginal_likelihood_derivatives_probit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

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
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison the result from GPML package with the minfunc function:
	/*cov =
		-0.034178423415816
		0.108245557597861
	*/ 
	abs_tolorance = get_abs_tolorance_lbfgs(-0.034178423415816, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.034178423415816, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.108245557597861, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, 0.108245557597861, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_posterior_mean_probit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);

	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);


	// comparison the result from GPML package with the minfunc function:
	/*post_mean =
		-0.505266873736866
		0.511503478056012
		0.506092360239034
		0.510734359252274
		-0.506072142343126
	*/
	SGVector<float64_t> approx_mean=inf->get_posterior_mean();
	abs_tolorance = get_abs_tolorance_lbfgs(-0.505266873736866, rel_tolorance);
	EXPECT_NEAR(approx_mean[0], -0.505266873736866, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.511503478056012, rel_tolorance);
	EXPECT_NEAR(approx_mean[1], 0.511503478056012, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.506092360239034, rel_tolorance);
	EXPECT_NEAR(approx_mean[2], 0.506092360239034, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.510734359252274, rel_tolorance);
	EXPECT_NEAR(approx_mean[3], 0.510734359252274, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.506072142343126, rel_tolorance);
	EXPECT_NEAR(approx_mean[4], -0.506072142343126, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_posterior_covariance_probit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	SGMatrix<float64_t> approx_cov=inf->get_posterior_covariance();

	// comparison the result from GPML package with the minfunc function:
	/*post_cov =
		0.661201597203589   -0.000005390820393    0.000004452771639    0.001055214193939    0.000025118418563
		-0.000005390820393    0.661904520588521   -0.000000300481742    0.007166673826843    0.000002019329325
		0.000004452771639   -0.000000300481742    0.661300448052085    0.000054316971105   -0.000000000087921
		0.001055214193939    0.007166673826843    0.000054316971105    0.661812036831663    0.000000009174084
		0.000025118418563    0.000002019329325   -0.000000000087921    0.000000009174084    0.661298049137338
	*/
	abs_tolorance = get_abs_tolorance_lbfgs(0.661201597203589, rel_tolorance);
	EXPECT_NEAR(approx_cov(0,0), 0.661201597203589, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.000005390820393, rel_tolorance);
	EXPECT_NEAR(approx_cov(0,1), -0.000005390820393, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000004452771639, rel_tolorance);
	EXPECT_NEAR(approx_cov(0,2), 0.000004452771639, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.001055214193939, rel_tolorance);
	EXPECT_NEAR(approx_cov(0,3), 0.001055214193939, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000025118418563, rel_tolorance);
	EXPECT_NEAR(approx_cov(0,4), 0.000025118418563, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(0.661904520588521, rel_tolorance);
	EXPECT_NEAR(approx_cov(1,1), 0.661904520588521, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.000000300481742, rel_tolorance);
	EXPECT_NEAR(approx_cov(1,2), -0.000000300481742, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.007166673826843, rel_tolorance);
	EXPECT_NEAR(approx_cov(1,3), 0.007166673826843, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000002019329325, rel_tolorance);
	EXPECT_NEAR(approx_cov(1,4), 0.000002019329325, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(0.661300448052085, rel_tolorance);
	EXPECT_NEAR(approx_cov(2,2), 0.661300448052085, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000054316971105, rel_tolorance);
	EXPECT_NEAR(approx_cov(2,3), 0.000054316971105, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.000000000087921, rel_tolorance);
	EXPECT_NEAR(approx_cov(2,4), -0.000000000087921, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(0.661812036831663, rel_tolorance);
	EXPECT_NEAR(approx_cov(3,3), 0.661812036831663, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.000000009174084, rel_tolorance);
	EXPECT_NEAR(approx_cov(3,4), 0.000000009174084, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(0.661298049137338, rel_tolorance);
	EXPECT_NEAR(approx_cov(4,4), 0.661298049137338, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_cholesky_logit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*L =
		1.116951738967970   0.035936999465803   0.044634190304333   0.091234784330075   0.076228914213078
		0.000000000000000   1.103969927331121   0.038656810617333   0.158332612953232   0.147925959238743
		0.000000000000000   0.000000000000000   1.114697377759363   0.090486956943561   0.014201836455373
		0.000000000000000   0.000000000000000   0.000000000000000   1.092971984807804   0.113570818492124
		0.000000000000000   0.000000000000000   0.000000000000000   0.000000000000000   1.088750747273819
	*/
	SGMatrix<float64_t> L=inf->get_cholesky();
	abs_tolorance = get_abs_tolorance_lbfgs(1.116951738967970, rel_tolorance);
	EXPECT_NEAR(L(0,0), 1.116951738967970, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.035936999465803, rel_tolorance);
	EXPECT_NEAR(L(0,1), 0.035936999465803, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.044634190304333, rel_tolorance);
	EXPECT_NEAR(L(0,2), 0.044634190304333, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.091234784330075, rel_tolorance);
	EXPECT_NEAR(L(0,3), 0.091234784330075, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.076228914213078, rel_tolorance);
	EXPECT_NEAR(L(0,4), 0.076228914213078, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.103969927331121, rel_tolorance);
	EXPECT_NEAR(L(1,1), 1.103969927331121, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.038656810617333, rel_tolorance);
	EXPECT_NEAR(L(1,2), 0.038656810617333, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.158332612953232, rel_tolorance);
	EXPECT_NEAR(L(1,3), 0.158332612953232, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.147925959238743, rel_tolorance);
	EXPECT_NEAR(L(1,4), 0.147925959238743, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.114697377759363, rel_tolorance);
	EXPECT_NEAR(L(2,2), 1.114697377759363, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.090486956943561, rel_tolorance);
	EXPECT_NEAR(L(2,3), 0.090486956943561, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.014201836455373, rel_tolorance);
	EXPECT_NEAR(L(2,4), 0.014201836455373, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.092971984807804, rel_tolorance);
	EXPECT_NEAR(L(3,3), 1.092971984807804, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.113570818492124, rel_tolorance);
	EXPECT_NEAR(L(3,4), 0.113570818492124, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.088750747273819, rel_tolorance);
	EXPECT_NEAR(L(4,4), 1.088750747273819, abs_tolorance);


	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_alpha_logit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);

	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	SGVector<float64_t> alpha=inf->get_alpha();
	// comparison the result from GPML package with the minfunc function:
	/*alpha =
		0.450818570957885
		-0.326913504639893
		0.437046061211033
		-0.382393225390073
		-0.345634045169136
	*/
	abs_tolorance = get_abs_tolorance_lbfgs(0.450818570957885, rel_tolorance);
	EXPECT_NEAR(alpha[0], 0.450818570957885, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.326913504639893, rel_tolorance);
	EXPECT_NEAR(alpha[1], -0.326913504639893, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.437046061211033, rel_tolorance);
	EXPECT_NEAR(alpha[2], 0.437046061211033, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.382393225390073, rel_tolorance);
	EXPECT_NEAR(alpha[3], -0.382393225390073, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.345634045169136, rel_tolorance);
	EXPECT_NEAR(alpha[4], -0.345634045169136, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_negative_marginal_likelihood_logit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*nlZ =
		3.387608216855656
	*/
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	abs_tolorance = get_abs_tolorance_lbfgs(3.387608216855656, rel_tolorance);
	EXPECT_NEAR(nml, 3.387608216855656, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_marginal_likelihood_derivatives_logit_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

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
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison the result from GPML package with the minfunc function:
	/*cov =
		0.266463865609896
		-0.068636643738048
	*/
	abs_tolorance = get_abs_tolorance_lbfgs(0.266463865609896, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, 0.266463865609896, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.068636643738048, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.068636643738048, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_cholesky_gaussian_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*L =
		1.414213562373095   0.492949892971257   0.433406478663272   0.323461940381075   0.019293928650029
		0.000000000000000   1.325518918393708   0.585882848653472   0.575780055183854   0.133086861715402
		0.000000000000000   0.000000000000000   1.211981894215584   0.403409348006210   0.125152120934987
		0.000000000000000   0.000000000000000   0.000000000000000   1.183685177367104   0.189875111083568
		0.000000000000000   0.000000000000000   0.000000000000000   0.000000000000000   1.389316385987177
	*/

	SGMatrix<float64_t> L=inf->get_cholesky();
	abs_tolorance = get_abs_tolorance_lbfgs(1.414213562373095, rel_tolorance);
	EXPECT_NEAR(L(0,0), 1.414213562373095, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.492949892971257, rel_tolorance);
	EXPECT_NEAR(L(0,1), 0.492949892971257, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.433406478663272, rel_tolorance);
	EXPECT_NEAR(L(0,2), 0.433406478663272, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.323461940381075, rel_tolorance);
	EXPECT_NEAR(L(0,3), 0.323461940381075, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.019293928650029, rel_tolorance);
	EXPECT_NEAR(L(0,4), 0.019293928650029, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.325518918393708, rel_tolorance);
	EXPECT_NEAR(L(1,1), 1.325518918393708, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.585882848653472, rel_tolorance);
	EXPECT_NEAR(L(1,2), 0.585882848653472, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.575780055183854, rel_tolorance);
	EXPECT_NEAR(L(1,3), 0.575780055183854, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.133086861715402, rel_tolorance);
	EXPECT_NEAR(L(1,4), 0.133086861715402, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.211981894215584, rel_tolorance);
	EXPECT_NEAR(L(2,2), 1.211981894215584, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.403409348006210, rel_tolorance);
	EXPECT_NEAR(L(2,3), 0.403409348006210, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.125152120934987, rel_tolorance);
	EXPECT_NEAR(L(2,4), 0.125152120934987, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.183685177367104, rel_tolorance);
	EXPECT_NEAR(L(3,3), 1.183685177367104, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.189875111083568, rel_tolorance);
	EXPECT_NEAR(L(3,4), 0.189875111083568, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.389316385987177, rel_tolorance);
	EXPECT_NEAR(L(4,4), 1.389316385987177, abs_tolorance);


	// clean up
	SG_UNREF(inf);
}
TEST(LaplacianInferenceMethodWithLBFGS,get_alpha_gaussian_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*alpha = 
		0.112589537139413
		0.030951587759558
		0.265522614808735
		0.372392096573089
		0.660353604050175
	*/
	SGVector<float64_t> alpha=inf->get_alpha();
	abs_tolorance = get_abs_tolorance_lbfgs(0.112589537139413, rel_tolorance);
	EXPECT_NEAR(alpha[0], 0.112589537139413, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.030951587759558, rel_tolorance);
	EXPECT_NEAR(alpha[1], 0.030951587759558, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.265522614808735, rel_tolorance);
	EXPECT_NEAR(alpha[2], 0.265522614808735, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.372392096573089, rel_tolorance);
	EXPECT_NEAR(alpha[3], 0.372392096573089, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.660353604050175, rel_tolorance);
	EXPECT_NEAR(alpha[4], 0.660353604050175, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_negative_marginal_likelihood_gaussian_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*nlZ =
		6.861543230523298
	*/
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	abs_tolorance = get_abs_tolorance_lbfgs(6.861543230523298, rel_tolorance);
	EXPECT_NEAR(nml, 6.861543230523298, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_marginal_likelihood_derivatives_gaussian_likelihood)
{

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;
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

	float64_t ell=0.1;

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			lik);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

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

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison the result from GPML package with the minfunc function:
	/*cov = 
		-0.851031385976160
		-0.570516239076101]
		lik =
		0.007407293825117
	*/ 
	abs_tolorance = get_abs_tolorance_lbfgs(0.007407293825117, rel_tolorance);
	EXPECT_NEAR(dnlZ_lik, 0.007407293825117, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.851031385976160, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.851031385976160, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.570516239076101, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.570516239076101, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}


TEST(LaplacianInferenceMethodWithLBFGS,get_cholesky_t_likelihood)
{

	float64_t rel_tolorance = 1e-1;
	float64_t abs_tolorance;
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
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/* L =
		1.523700622513149   0.607340583120219   0.521037907065038   0.378596005447475   0.019953356735640
		0.000000000000000   1.401561489176888   0.693360672949196   0.669737448715427   0.139806033095216
		0.000000000000000   0.000000000000000   1.231732036088494   0.432550803635198   0.123874068633327
		0.000000000000000   0.000000000000000   0.000000000000000   1.193424304765556   0.189334517311970
		0.000000000000000   0.000000000000000   0.000000000000000   0.000000000000000   1.366835295900706
	*/
	SGMatrix<float64_t> L=inf->get_cholesky();

	abs_tolorance = get_abs_tolorance_lbfgs(1.523700622513149, rel_tolorance);
	EXPECT_NEAR(L(0,0), 1.523700622513149, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.607340583120219, rel_tolorance);
	EXPECT_NEAR(L(0,1), 0.607340583120219, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.521037907065038, rel_tolorance);
	EXPECT_NEAR(L(0,2), 0.521037907065038, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.378596005447475, rel_tolorance);
	EXPECT_NEAR(L(0,3), 0.378596005447475, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.019953356735640, rel_tolorance);
	EXPECT_NEAR(L(0,4), 0.019953356735640, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.401561489176888, rel_tolorance);
	EXPECT_NEAR(L(1,1), 1.401561489176888, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.693360672949196, rel_tolorance);
	EXPECT_NEAR(L(1,2), 0.693360672949196, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.669737448715427, rel_tolorance);
	EXPECT_NEAR(L(1,3), 0.669737448715427, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.139806033095216, rel_tolorance);
	EXPECT_NEAR(L(1,4), 0.139806033095216, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.231732036088494, rel_tolorance);
	EXPECT_NEAR(L(2,2), 1.231732036088494, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.432550803635198, rel_tolorance);
	EXPECT_NEAR(L(2,3), 0.432550803635198, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.123874068633327, rel_tolorance);
	EXPECT_NEAR(L(2,4), 0.123874068633327, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.193424304765556, rel_tolorance);
	EXPECT_NEAR(L(3,3), 1.193424304765556, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.189334517311970, rel_tolorance);
	EXPECT_NEAR(L(3,4), 0.189334517311970, abs_tolorance);

	abs_tolorance = get_abs_tolorance_lbfgs(1.366835295900706, rel_tolorance);
	EXPECT_NEAR(L(4,4), 1.366835295900706, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_alpha_t_likelihood)
{

	float64_t rel_tolorance = 1e-1;
	float64_t abs_tolorance;
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
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf 
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*alpha = 
		0.124677478636837
		-0.011322148653691
		0.291185918072183
		0.414106934704980
		0.710852596742621
	*/
	SGVector<float64_t> alpha=inf->get_alpha();
	abs_tolorance = get_abs_tolorance_lbfgs(0.124677478636837, rel_tolorance);
	EXPECT_NEAR(alpha[0], 0.124677478636837, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.011322148653691, rel_tolorance);
	EXPECT_NEAR(alpha[1], -0.011322148653691, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.291185918072183, rel_tolorance);
	EXPECT_NEAR(alpha[2], 0.291185918072183, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.414106934704980, rel_tolorance);
	EXPECT_NEAR(alpha[3], 0.414106934704980, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(0.710852596742621, rel_tolorance);
	EXPECT_NEAR(alpha[4], 0.710852596742621, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_negative_marginal_likelihood_t_likelihood)
{

	float64_t rel_tolorance = 1e-1;
	float64_t abs_tolorance;
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
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf 
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			likelihood);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

	// comparison the result from GPML package with the minfunc function:
	/*nlZ =
		7.489169113992463
	*/
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	abs_tolorance = get_abs_tolorance_lbfgs(7.489169113992463, rel_tolorance);
	EXPECT_NEAR(nml, 7.489169113992463, abs_tolorance);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethodWithLBFGS,get_marginal_likelihood_derivatives_t_likelihood)
{

	float64_t rel_tolorance = 1e-1;
	float64_t abs_tolorance;
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
	CStudentsTLikelihood* lik=new CStudentsTLikelihood(0.25, 3);

	// specify GP regression with exact inference
	CLaplacianInferenceMethodWithLBFGS* inf
		= new CLaplacianInferenceMethodWithLBFGS(kernel,
			features_train,
			mean,
			labels_train,
			lik);
	int m = 100;
	int max_linesearch = 1000;
	int linesearch = 0;
	int max_iterations = 1000;
	float64_t delta = 1e-15;
	int past = 0;
	float64_t epsilon = 1e-15;
	bool enable_newton_if_fail = false;
	inf->set_lbfgs_parameters(m, 
		max_linesearch,
		linesearch,
		max_iterations,
		delta, 
		past, 
		epsilon,
		enable_newton_if_fail
		);

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

	// comparison the result from GPML package with the minfunc function:
	/*cov =
		-0.843641535114105
		-0.301771081861900
		lik =
		-0.649318379107740
		-0.155672464565009
	*/ 
	abs_tolorance = get_abs_tolorance_lbfgs(-0.649318379107740, rel_tolorance);
	EXPECT_NEAR(dnlZ_df, -0.649318379107740, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.155672464565009, rel_tolorance);
	EXPECT_NEAR(dnlZ_sigma, -0.155672464565009, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.843641535114105, rel_tolorance);
	EXPECT_NEAR(dnlZ_ell, -0.843641535114105, abs_tolorance);
	abs_tolorance = get_abs_tolorance_lbfgs(-0.301771081861900, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.301771081861900, abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}
#endif /* HAVE_EIGEN3 */
