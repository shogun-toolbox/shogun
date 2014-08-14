/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

// Eigen3 is required for working with this example
#ifdef HAVE_EIGEN3

#include <shogun/base/init.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/SingleLaplacianInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/classifier/GaussianProcessClassification.h>
#include <shogun/io/CSVFile.h>

using namespace shogun;

// files with training data
const char* fname_feat_train="../data/fm_train_real.dat";
const char* fname_label_train="../data/label_train_twoclass.dat";

// file with testing data
const char* fname_feat_test="../data/fm_test_real.dat";

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	// trainig data
	SGMatrix<float64_t> X_train;
	SGVector<float64_t> y_train;

	// load training features from file
	CCSVFile* file_feat_train=new CCSVFile(fname_feat_train);
	X_train.load(file_feat_train);
	SG_UNREF(file_feat_train);

	// load training labels from file
	CCSVFile* file_label_train=new CCSVFile(fname_label_train);
	y_train.load(file_label_train);
	SG_UNREF(file_label_train);

	// testing features
	SGMatrix<float64_t> X_test;

	// load testing features from file
	CCSVFile* file_feat_test=new CCSVFile(fname_feat_test);
	X_test.load(file_feat_test);
	SG_UNREF(file_feat_test);

	// convert training and testing data into shogun representation
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X_train);
	CBinaryLabels* lab_train=new CBinaryLabels(y_train);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	SG_REF(feat_test);

	// create Gaussian kernel with width = 2.0
	CGaussianKernel* kernel=new CGaussianKernel(10, 2.0);

	// create zero mean function
	CZeroMean* mean=new CZeroMean();

	// you can easily switch between probit and logit likelihood models
    // by uncommenting/commenting the following lines:

	// create probit likelihood model
	// CProbitLikelihood* lik=new CProbitLikelihood();

	// create logit likelihood model
	CLogitLikelihood* lik=new CLogitLikelihood();

	// you can easily switch between SingleLaplace and EP approximation by
	// uncommenting/commenting the following lines:

	// specify SingleLaplace approximation inference method
	// CSingleLaplacianInferenceMethod* inf=new CSingleLaplacianInferenceMethod(kernel,
	//		feat_train, mean, lab_train, lik);

	// specify EP approximation inference method
	CEPInferenceMethod* inf=new CEPInferenceMethod(kernel, feat_train, mean,
			lab_train, lik);

	// create and train GP classifier, which uses SingleLaplace approximation
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// apply binary classification to the test data and get -1/+1
	// labels of the predictions
	CBinaryLabels* predictions=gpc->apply_binary(feat_test);
	predictions->get_labels().display_vector("predictions");

	// get probabilities p(y*=1|x*) for each testing feature x*
	SGVector<float64_t> p_test=gpc->get_probabilities(feat_test);
	p_test.display_vector("predictive probability");

	// get predictive mean
	SGVector<float64_t> mu_test=gpc->get_mean_vector(feat_test);
	mu_test.display_vector("predictive mean");

	// get predictive variance
	SGVector<float64_t> s2_test=gpc->get_variance_vector(feat_test);
	s2_test.display_vector("predictive variance");

	// free up memory
	SG_UNREF(gpc);
	SG_UNREF(predictions);
	SG_UNREF(feat_test);

	exit_shogun();
	return 0;
}

#else

int main(int argc, char **argv)
{
	return 0;
}

#endif /* HAVE_EIGEN3 */
