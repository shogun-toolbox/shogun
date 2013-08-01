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
#include <shogun/machine/gp/LaplacianInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/classifier/GaussianProcessBinaryClassification.h>

using namespace shogun;

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	// create some toy classification data with n training examples
	// and m testing features
	const index_t n=10, m=25;

	SGMatrix<float64_t> X_train(2, n);
	SGVector<float64_t> y_train(n);
	SGMatrix<float64_t> X_test(2, m);

	// fill matrix of training features with random nonsense
	X_train(0, 0)=0.0919736;
	X_train(0, 1)=-0.3813827;
	X_train(0, 2)=-1.8011128;
	X_train(0, 3)=-1.4603061;
	X_train(0, 4)=-0.1386884;
	X_train(0, 5)=0.7827657;
	X_train(0, 6)=-0.1369808;
	X_train(0, 7)=0.0058596;
	X_train(0, 8)=0.1059573;
	X_train(0, 9)=-1.3059609;

	X_train(1, 0)=1.4186892;
	X_train(1, 1)=0.2271813;
	X_train(1, 2)=0.3451326;
	X_train(1, 3)=0.4495962;
	X_train(1, 4)=1.2066144;
	X_train(1, 5)=-0.5425118;
	X_train(1, 6)=1.3479000;
	X_train(1, 7)=0.7181545;
	X_train(1, 8)=0.4036014;
	X_train(1, 9)=0.8928408;

	// fill vector of training labels with -1/+1 random values
	y_train[0]=1.0;
	y_train[1]=-1.0;
	y_train[2]=-1.0;
	y_train[3]=-1.0;
	y_train[4]=-1.0;
	y_train[5]=1.0;
	y_train[6]=-1.0;
	y_train[7]=1.0;
	y_train[8]=1.0;
	y_train[9]=-1.0;

	// fill matrix of testing features
	index_t i=0;

	for (index_t x1=-2; x1<=2; x1++)
		for (index_t x2=-2; x2<=2; x2++)
		{
			X_test(0, i)=(float64_t)x1;
			X_test(1, i)=(float64_t)x2;
			i++;
		}

	// convert training and testing data into shogun representation
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X_train);
	CBinaryLabels* lab_train=new CBinaryLabels(y_train);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);

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

	// specify Laplace approximation inference method
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		feat_train, mean, lab_train, lik);

	// create and train GP classifier, which uses Laplace approximation
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
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

	exit_shogun();
	return 0;
}

#else

int main(int argc, char **argv)
{
	return 0;
}

#endif /* HAVE_EIGEN3 */
