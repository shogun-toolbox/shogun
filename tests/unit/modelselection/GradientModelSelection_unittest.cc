/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#if defined HAVE_NLOPT

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

#include <gtest/gtest.h>

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
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X_train);
	CRegressionLabels* lab_train=new CRegressionLabels(y_train);

	// choose Gaussian kernel with width=2*ell^2, where ell=3
	CGaussianKernel* kernel=new CGaussianKernel(10, 3*3*2.0);

	// create zero mean
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma
	CGaussianLikelihood* lik=new CGaussianLikelihood(4.0);

	// specify exact inference method
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, lab_train, lik);

	// set kernel scale
	inf->set_scale(3);

	// specify GP regression with exact inference
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// specify gradient evaluation object
	CGradientEvaluation* grad_eval=new CGradientEvaluation(gpr, feat_train,
			lab_train, new CGradientCriterion(), false);

	// set diffirentiable function
	grad_eval->set_function(inf);

	// specify gradient search
	CGradientModelSelection* grad_search=new CGradientModelSelection(grad_eval);

	// find best parameter combination
	CParameterCombination* best_comb=grad_search->select_model(false);
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
	EXPECT_NEAR(CMath::sqrt(width/2.0), 2.024939917854716, 1E-3);
	EXPECT_NEAR(sigma, 1.862122012902403, 1E-3);

	// cleanup
	SG_UNREF(grad_search);
	SG_UNREF(best_comb);
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
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with EP inference and kernel scale=1.5
	CEPInferenceMethod* inf=new CEPInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// specify GP classification with EP inference
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);

	// specify gradient evaluation object
	CGradientEvaluation* grad_eval=new CGradientEvaluation(gpc, features_train,
			labels_train, new CGradientCriterion(), false);

	// set diffirentiable function
	grad_eval->set_function(inf);

	// specify gradient search
	CGradientModelSelection* grad_search=new CGradientModelSelection(grad_eval);

	// find best parameter combination
	CParameterCombination* best_comb=grad_search->select_model(false);
	best_comb->apply_to_machine(gpc);

	// compare negative log marginal likelihood with result from GPML toolbox
	float64_t nlZ=inf->get_negative_log_marginal_likelihood();
	EXPECT_NEAR(nlZ, 3.334009, 1E-3);

	// cleanup
	SG_UNREF(grad_search);
	SG_UNREF(best_comb);
}

#endif /* defined HAVE_NLOPT */
