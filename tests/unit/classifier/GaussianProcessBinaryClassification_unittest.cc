/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/machine/gp/LaplacianInferenceMethod.h>
#include <shogun/classifier/GaussianProcessBinaryClassification.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(GaussianProcessBinaryClassification,get_mean_vector)
{
	// create some easy random classification data
	index_t n=10, m=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m);

	feat_train(0, 0)=0.0919736;
	feat_train(0, 1)=-0.3813827;
	feat_train(0, 2)=-1.8011128;
	feat_train(0, 3)=-1.4603061;
	feat_train(0, 4)=-0.1386884;
	feat_train(0, 5)=0.7827657;
	feat_train(0, 6)=-0.1369808;
	feat_train(0, 7)=0.0058596;
	feat_train(0, 8)=0.1059573;
	feat_train(0, 9)=-1.3059609;

	feat_train(1, 0)=1.4186892;
	feat_train(1, 1)=0.2271813;
	feat_train(1, 2)=0.3451326;
	feat_train(1, 3)=0.4495962;
	feat_train(1, 4)=1.2066144;
	feat_train(1, 5)=-0.5425118;
	feat_train(1, 6)=1.3479000;
	feat_train(1, 7)=0.7181545;
	feat_train(1, 8)=0.4036014;
	feat_train(1, 9)=0.8928408;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=-1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;
	lab_train[5]=1.0;
	lab_train[6]=-1.0;
	lab_train[7]=1.0;
	lab_train[8]=1.0;
	lab_train[9]=-1.0;

	// create test features
	for (index_t x1=-2; x1<=2; x1++)
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
	gpc->train();

	// compare mean vector with result form GPML
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	EXPECT_NEAR(mean_vector[0], -0.023547, 1E-6);
	EXPECT_NEAR(mean_vector[1], -0.164421, 1E-6);
	EXPECT_NEAR(mean_vector[2], -0.447812, 1E-6);
	EXPECT_NEAR(mean_vector[3], -0.472429, 1E-6);
	EXPECT_NEAR(mean_vector[4], -0.205391, 1E-6);
	EXPECT_NEAR(mean_vector[5], -0.011335, 1E-6);
	EXPECT_NEAR(mean_vector[6], -0.131013, 1E-6);
	EXPECT_NEAR(mean_vector[7], -0.427260, 1E-6);
	EXPECT_NEAR(mean_vector[8], -0.527281, 1E-6);
	EXPECT_NEAR(mean_vector[9], -0.274684, 1E-6);
	EXPECT_NEAR(mean_vector[10], 0.055529, 1E-6);
	EXPECT_NEAR(mean_vector[11], 0.152024, 1E-6);
	EXPECT_NEAR(mean_vector[12], 0.174282, 1E-6);
	EXPECT_NEAR(mean_vector[13], 0.010823, 1E-6);
	EXPECT_NEAR(mean_vector[14], -0.072773, 1E-6);
	EXPECT_NEAR(mean_vector[15], 0.090192, 1E-6);
	EXPECT_NEAR(mean_vector[16], 0.288418, 1E-6);
	EXPECT_NEAR(mean_vector[17], 0.409275, 1E-6);
	EXPECT_NEAR(mean_vector[18], 0.281221, 1E-6);
	EXPECT_NEAR(mean_vector[19], 0.088383, 1E-6);
	EXPECT_NEAR(mean_vector[20], 0.043796, 1E-6);
	EXPECT_NEAR(mean_vector[21], 0.130462, 1E-6);
	EXPECT_NEAR(mean_vector[22], 0.170565, 1E-6);
	EXPECT_NEAR(mean_vector[23], 0.113007, 1E-6);
	EXPECT_NEAR(mean_vector[24], 0.041654, 1E-6);

	SG_UNREF(gpc);
}

TEST(GaussianProcessBinaryClassification,get_variance_vector)
{
	// create some easy random classification data
	index_t n=10, m=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m);

	feat_train(0, 0)=0.0919736;
	feat_train(0, 1)=-0.3813827;
	feat_train(0, 2)=-1.8011128;
	feat_train(0, 3)=-1.4603061;
	feat_train(0, 4)=-0.1386884;
	feat_train(0, 5)=0.7827657;
	feat_train(0, 6)=-0.1369808;
	feat_train(0, 7)=0.0058596;
	feat_train(0, 8)=0.1059573;
	feat_train(0, 9)=-1.3059609;

	feat_train(1, 0)=1.4186892;
	feat_train(1, 1)=0.2271813;
	feat_train(1, 2)=0.3451326;
	feat_train(1, 3)=0.4495962;
	feat_train(1, 4)=1.2066144;
	feat_train(1, 5)=-0.5425118;
	feat_train(1, 6)=1.3479000;
	feat_train(1, 7)=0.7181545;
	feat_train(1, 8)=0.4036014;
	feat_train(1, 9)=0.8928408;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=-1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;
	lab_train[5]=1.0;
	lab_train[6]=-1.0;
	lab_train[7]=1.0;
	lab_train[8]=1.0;
	lab_train[9]=-1.0;

	// create test features
	for (index_t x1=-2; x1<=2; x1++)
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train gaussian process classifier
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
	gpc->train();

	// compare variance vector with result form GPML
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);

	EXPECT_NEAR(variance_vector[0], 0.99945, 1E-5);
	EXPECT_NEAR(variance_vector[1], 0.97297, 1E-5);
	EXPECT_NEAR(variance_vector[2], 0.79946, 1E-5);
	EXPECT_NEAR(variance_vector[3], 0.77681, 1E-5);
	EXPECT_NEAR(variance_vector[4], 0.95781, 1E-5);
	EXPECT_NEAR(variance_vector[5], 0.99987, 1E-5);
	EXPECT_NEAR(variance_vector[6], 0.98284, 1E-5);
	EXPECT_NEAR(variance_vector[7], 0.81745, 1E-5);
	EXPECT_NEAR(variance_vector[8], 0.72197, 1E-5);
	EXPECT_NEAR(variance_vector[9], 0.92455, 1E-5);
	EXPECT_NEAR(variance_vector[10], 0.99692, 1E-5);
	EXPECT_NEAR(variance_vector[11], 0.97689, 1E-5);
	EXPECT_NEAR(variance_vector[12], 0.96963, 1E-5);
	EXPECT_NEAR(variance_vector[13], 0.99988, 1E-5);
	EXPECT_NEAR(variance_vector[14], 0.99470, 1E-5);
	EXPECT_NEAR(variance_vector[15], 0.99187, 1E-5);
	EXPECT_NEAR(variance_vector[16], 0.91682, 1E-5);
	EXPECT_NEAR(variance_vector[17], 0.83249, 1E-5);
	EXPECT_NEAR(variance_vector[18], 0.92091, 1E-5);
	EXPECT_NEAR(variance_vector[19], 0.99219, 1E-5);
	EXPECT_NEAR(variance_vector[20], 0.99808, 1E-5);
	EXPECT_NEAR(variance_vector[21], 0.98298, 1E-5);
	EXPECT_NEAR(variance_vector[22], 0.97091, 1E-5);
	EXPECT_NEAR(variance_vector[23], 0.98723, 1E-5);
	EXPECT_NEAR(variance_vector[24], 0.99826, 1E-5);

	SG_UNREF(gpc);
}

TEST(GaussianProcessBinaryClassification,get_probabilities)
{
	// create some easy random classification data
	index_t n=10, m=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m);

	feat_train(0, 0)=0.0919736;
	feat_train(0, 1)=-0.3813827;
	feat_train(0, 2)=-1.8011128;
	feat_train(0, 3)=-1.4603061;
	feat_train(0, 4)=-0.1386884;
	feat_train(0, 5)=0.7827657;
	feat_train(0, 6)=-0.1369808;
	feat_train(0, 7)=0.0058596;
	feat_train(0, 8)=0.1059573;
	feat_train(0, 9)=-1.3059609;

	feat_train(1, 0)=1.4186892;
	feat_train(1, 1)=0.2271813;
	feat_train(1, 2)=0.3451326;
	feat_train(1, 3)=0.4495962;
	feat_train(1, 4)=1.2066144;
	feat_train(1, 5)=-0.5425118;
	feat_train(1, 6)=1.3479000;
	feat_train(1, 7)=0.7181545;
	feat_train(1, 8)=0.4036014;
	feat_train(1, 9)=0.8928408;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=-1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;
	lab_train[5]=1.0;
	lab_train[6]=-1.0;
	lab_train[7]=1.0;
	lab_train[8]=1.0;
	lab_train[9]=-1.0;

	// create test features
	for (index_t x1=-2; x1<=2; x1++)
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train gaussian process classifier
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
	gpc->train();

	// compare variance vector with result form GPML
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);

	EXPECT_NEAR(probabilities[0], 0.48823, 1E-5);
	EXPECT_NEAR(probabilities[1], 0.41779, 1E-5);
	EXPECT_NEAR(probabilities[2], 0.27609, 1E-5);
	EXPECT_NEAR(probabilities[3], 0.26379, 1E-5);
	EXPECT_NEAR(probabilities[4], 0.39730, 1E-5);
	EXPECT_NEAR(probabilities[5], 0.49433, 1E-5);
	EXPECT_NEAR(probabilities[6], 0.43449, 1E-5);
	EXPECT_NEAR(probabilities[7], 0.28637, 1E-5);
	EXPECT_NEAR(probabilities[8], 0.23636,1E-5);
	EXPECT_NEAR(probabilities[9], 0.36266, 1E-5);
	EXPECT_NEAR(probabilities[10], 0.52776, 1E-5);
	EXPECT_NEAR(probabilities[11], 0.57601, 1E-5);
	EXPECT_NEAR(probabilities[12], 0.58714, 1E-5);
	EXPECT_NEAR(probabilities[13], 0.50541, 1E-5);
	EXPECT_NEAR(probabilities[14], 0.46361, 1E-5);
	EXPECT_NEAR(probabilities[15], 0.54510, 1E-5);
	EXPECT_NEAR(probabilities[16], 0.64421, 1E-5);
	EXPECT_NEAR(probabilities[17], 0.70464, 1E-5);
	EXPECT_NEAR(probabilities[18], 0.64061, 1E-5);
	EXPECT_NEAR(probabilities[19], 0.54419, 1E-5);
	EXPECT_NEAR(probabilities[20], 0.52190, 1E-5);
	EXPECT_NEAR(probabilities[21], 0.56523, 1E-5);
	EXPECT_NEAR(probabilities[22], 0.58528, 1E-5);
	EXPECT_NEAR(probabilities[23], 0.55650, 1E-5);
	EXPECT_NEAR(probabilities[24], 0.52083, 1E-5);

	SG_UNREF(gpc);
}

#endif /* HAVE_EIGEN3 */
