/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wu Lin, Roman Votyakov, Heiko Strathmann, Grigorii Guz,
 *          Thoralf Klein, Pan Deng
 */


#include <gtest/gtest.h>
#include <shogun/lib/config.h>

#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/classifier/GaussianProcessClassification.h>

#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/mathematics/Math.h>
#include <shogun/optimization/NLOPTMinimizer.h>


using namespace shogun;

TEST(GradientModelSelection,select_model_exact_inference)
{
	index_t ntr=5;

	SGMatrix<float64_t> X_train(1, ntr);
	SGVector<float64_t> y_train(ntr);

	X_train[0]=1.25107;
	X_train[1]=-2.16097;
	X_train[2]=-0.00034;
	X_train[3]=4.90699;
	X_train[4]=-2.44026;

	y_train[0]=-2.39635;
	y_train[1]=-1.00358;
	y_train[2]=-6.18139;
	y_train[3]=5.35533;
	y_train[4]=-3.08232;

	// shogun representation of features and labels
	auto feat_train=std::make_shared<DenseFeatures<float64_t>>(X_train);
	auto lab_train=std::make_shared<RegressionLabels>(y_train);

	// choose Gaussian kernel with width=2*ell^2, where ell=3
	auto kernel=std::make_shared<GaussianKernel>(10, 3*3*2.0);

	// create zero mean
	auto mean=std::make_shared<ZeroMean>();

	// Gaussian likelihood with sigma
	auto lik=std::make_shared<GaussianLikelihood>(4.0);

	// specify exact inference method
	auto inf=std::make_shared<ExactInferenceMethod>(kernel, feat_train,
			mean, lab_train, lik);

	// set kernel scale
	inf->set_scale(3);

	// specify GP regression with exact inference
	auto gpr=std::make_shared<GaussianProcessRegression>(inf);

	// specify gradient evaluation object
	auto grad_eval=std::make_shared<GradientEvaluation>(gpr, feat_train,
			lab_train, std::make_shared<GradientCriterion>(), false);

	// set diffirentiable function
	grad_eval->set_function(inf);

	// specify gradient search
	auto grad_search=std::make_shared<GradientModelSelection>(grad_eval);

	auto minimizer = std::make_shared<NLOPTMinimizer>();
	minimizer->set_nlopt_parameters(LD_MMA);
	grad_search->set_minimizer(minimizer);

	// find best parameter combination
	auto best_comb=grad_search->select_model(false);
	best_comb->apply_to_machine(gpr);

	// compare negative log marginal likelihood with result from GPML toolbox
	// Note that the result is based on input (there are many local mode results)
	//nlz= 1.334133e+01
	//exp(hyp.lik) = 1.862122012902403
	//exp(hyp.cov) = 2.024939917854716   3.778912324292096
	float64_t nlZ=inf->get_negative_log_marginal_likelihood();
	EXPECT_NEAR(nlZ, 13.34133, 1E-4);

	// get hyperparameters
	float64_t scale=inf->get_scale();
	float64_t width=kernel->get_width();
	float64_t sigma=lik->get_sigma();

	// compare hyperparameters with result from GPML toolbox
	EXPECT_NEAR(scale, 3.778912324292096, 1E-3);
	EXPECT_NEAR(std::sqrt(width / 2.0), 2.024939917854716, 1E-3);
	EXPECT_NEAR(sigma, 1.862122012902403, 1E-3);

}


TEST(GradientModelSelection,select_model_ep_inference)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// specify GP classification with EP inference
	auto gpc=std::make_shared<GaussianProcessClassification>(inf);

	// specify gradient evaluation object
	auto grad_eval=std::make_shared<GradientEvaluation>(gpc, features_train,
			labels_train, std::make_shared<GradientCriterion>(), false);

	// set diffirentiable function
	grad_eval->set_function(inf);

	// specify gradient search
	auto grad_search=std::make_shared<GradientModelSelection>(grad_eval);
	auto minimizer = std::make_shared<NLOPTMinimizer>();
	minimizer->set_nlopt_parameters(LD_MMA);
	grad_search->set_minimizer(minimizer);

	// find best parameter combination
	auto best_comb=grad_search->select_model(false);
	best_comb->apply_to_machine(gpc);

	// compare negative log marginal likelihood with result from GPML toolbox
	float64_t nlZ=inf->get_negative_log_marginal_likelihood();
	EXPECT_NEAR(nlZ, 3.334009, 1E-3);

}

#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN
