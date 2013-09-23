/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#if defined HAVE_EIGEN3 && defined HAVE_NLOPT

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/evaluation/GradientResult.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/classifier/GaussianProcessBinaryClassification.h>

#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/modelselection/GradientModelSelection.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(GradientModelSelection,select_model_exact_inference)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5;

	SGMatrix<float64_t> X_train(1, ntr);
	SGVector<float64_t> y_train(ntr);

	X_train[0]=1.25107;
	X_train[1]=2.16097;
	X_train[2]=0.00034;
	X_train[3]=0.90699;
	X_train[4]=0.44026;

	y_train[0]=0.39635;
	y_train[1]=0.00358;
	y_train[2]=-1.18139;
	y_train[3]=1.35533;
	y_train[4]=-0.08232;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X_train);
	CRegressionLabels* lab_train=new CRegressionLabels(y_train);

	// choose Gaussian kernel with width=2*ell^2=0.02, where ell=0.1
	CGaussianKernel* kernel=new CGaussianKernel(10, 0.02);

	// create zero mean
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma=0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify exact inference method
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, lab_train, lik);

	// set kernel scale
	inf->set_scale(1.8);

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
	float64_t nlZ=inf->get_negative_log_marginal_likelihood();
	EXPECT_NEAR(nlZ, 5.862416, 1E-6);

	// get hyperparameters
	float64_t scale=inf->get_scale();
	float64_t width=kernel->get_width();
	float64_t sigma=lik->get_sigma();

	// compare hyperparameters with result from GPML toolbox
	EXPECT_NEAR(scale, 0.833180109059697, 1E-2);
	EXPECT_NEAR(width, 0.190931901479123, 1E-2);
	EXPECT_NEAR(sigma, 4.25456324418582e-04, 1E-2);

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
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);

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

#endif /* defined HAVE_EIGEN3 && defined HAVE_NLOPT */
