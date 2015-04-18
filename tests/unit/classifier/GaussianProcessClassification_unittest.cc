/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (w) 2013 Roman Votyakov
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
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/machine/gp/GaussianARDFITCKernel.h>
#endif

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/SingleLaplacianInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/classifier/GaussianProcessClassification.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/SingleLaplacianInferenceMethodWithLBFGS.h>
#include <shogun/machine/gp/MultiLaplacianInferenceMethod.h>
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethod.h>
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethodWithLBFGS.h>

#include <shogun/machine/gp/KLCovarianceInferenceMethod.h>
#include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
#include <shogun/machine/gp/KLApproxDiagonalInferenceMethod.h>
#include <shogun/machine/gp/KLDualInferenceMethod.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/machine/gp/LogitDVGLikelihood.h>
#include <shogun/machine/gp/SoftMaxLikelihood.h>

using namespace shogun;

TEST(GaussianProcessClassification,get_mean_vector)
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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethod* inf=new CSingleLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
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

TEST(GaussianProcessClassification,get_variance_vector)
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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethod* inf=new CSingleLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
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

TEST(GaussianProcessClassification,get_probabilities)
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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethod* inf=new CSingleLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
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

TEST(GaussianProcessClassification,apply_preprocessor_and_binary)
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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CRescaleFeatures* preproc=new CRescaleFeatures();
	preproc->init(features_train);

	features_train->add_preprocessor(preproc);
	features_train->apply_preprocessor();

	features_test->add_preprocessor(preproc);
	features_test->apply_preprocessor();

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	CEPInferenceMethod* inf=new CEPInferenceMethod(kernel, features_train,
			mean, labels_train, likelihood);

	inf->set_scale(1.5);

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare predictions with result form GPML
	CBinaryLabels* prediction=gpc->apply_binary(features_test);

	SGVector<float64_t> p=prediction->get_labels();

	EXPECT_EQ(p[0], -1);
	EXPECT_EQ(p[1], -1);
	EXPECT_EQ(p[2], -1);
	EXPECT_EQ(p[3], -1);
	EXPECT_EQ(p[4], -1);
	EXPECT_EQ(p[5], -1);
	EXPECT_EQ(p[6], -1);
	EXPECT_EQ(p[7], -1);
	EXPECT_EQ(p[8], -1);
	EXPECT_EQ(p[9], -1);
	EXPECT_EQ(p[10], 1);
	EXPECT_EQ(p[11], 1);
	EXPECT_EQ(p[12], -1);
	EXPECT_EQ(p[13], -1);
	EXPECT_EQ(p[14], -1);
	EXPECT_EQ(p[15], 1);
	EXPECT_EQ(p[16], 1);
	EXPECT_EQ(p[17], 1);
	EXPECT_EQ(p[18], 1);
	EXPECT_EQ(p[19], 1);
	EXPECT_EQ(p[20], 1);
	EXPECT_EQ(p[21], 1);
	EXPECT_EQ(p[22], 1);
	EXPECT_EQ(p[23], 1);
	EXPECT_EQ(p[24], 1);

	SG_UNREF(gpc);
	SG_UNREF(prediction);
}

TEST(GaussianProcessClassificationUsingSingleLaplacianWithLBFGS,get_mean_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethodWithLBFGS* inf
	= new CSingleLaplacianInferenceMethodWithLBFGS(kernel,
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

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form GPML with the minfunc function
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	/*mean =
		-0.023547066779433
		-0.164420972889231
		-0.447812356229495
		-0.472428809447940
		-0.205391227282142
		-0.011335213830652
		-0.131012850981580
		-0.427259580375569
		-0.527281189501774
		-0.274684117023014
		0.055529455358847
		0.152023871056183
		0.174282413372574
		0.010823181344098
		-0.072772631266962
		0.090191676357209
		0.288417744414623
		0.409275122823904
		0.281220920795101
		0.088382525159406
		0.043796091667543
		0.130461505967524
		0.170564691797896
		0.113006930991411
		0.041654120309486
	*/

	abs_tolerance = CMath::get_abs_tolerance(-0.023547066779433, rel_tolerance);
	EXPECT_NEAR(mean_vector[0], -0.023547066779433, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.164420972889231, rel_tolerance);
	EXPECT_NEAR(mean_vector[1], -0.164420972889231, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.447812356229495, rel_tolerance);
	EXPECT_NEAR(mean_vector[2], -0.447812356229495, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.472428809447940, rel_tolerance);
	EXPECT_NEAR(mean_vector[3], -0.472428809447940, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.205391227282142, rel_tolerance);
	EXPECT_NEAR(mean_vector[4], -0.205391227282142, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.011335213830652, rel_tolerance);
	EXPECT_NEAR(mean_vector[5], -0.011335213830652, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.131012850981580, rel_tolerance);
	EXPECT_NEAR(mean_vector[6], -0.131012850981580, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.427259580375569, rel_tolerance);
	EXPECT_NEAR(mean_vector[7], -0.427259580375569, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.527281189501774, rel_tolerance);
	EXPECT_NEAR(mean_vector[8], -0.527281189501774, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.274684117023014, rel_tolerance);
	EXPECT_NEAR(mean_vector[9], -0.274684117023014, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.055529455358847, rel_tolerance);
	EXPECT_NEAR(mean_vector[10], 0.055529455358847, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.152023871056183, rel_tolerance);
	EXPECT_NEAR(mean_vector[11], 0.152023871056183, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.174282413372574, rel_tolerance);
	EXPECT_NEAR(mean_vector[12], 0.174282413372574, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.010823181344098, rel_tolerance);
	EXPECT_NEAR(mean_vector[13], 0.010823181344098, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.072772631266962, rel_tolerance);
	EXPECT_NEAR(mean_vector[14], -0.072772631266962, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.090191676357209, rel_tolerance);
	EXPECT_NEAR(mean_vector[15], 0.090191676357209, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.288417744414623, rel_tolerance);
	EXPECT_NEAR(mean_vector[16], 0.288417744414623, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.409275122823904, rel_tolerance);
	EXPECT_NEAR(mean_vector[17], 0.409275122823904, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.281220920795101, rel_tolerance);
	EXPECT_NEAR(mean_vector[18], 0.281220920795101, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.088382525159406, rel_tolerance);
	EXPECT_NEAR(mean_vector[19], 0.088382525159406, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.043796091667543, rel_tolerance);
	EXPECT_NEAR(mean_vector[20], 0.043796091667543, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.130461505967524, rel_tolerance);
	EXPECT_NEAR(mean_vector[21], 0.130461505967524, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.170564691797896, rel_tolerance);
	EXPECT_NEAR(mean_vector[22], 0.170564691797896, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.113006930991411, rel_tolerance);
	EXPECT_NEAR(mean_vector[23], 0.113006930991411, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.041654120309486, rel_tolerance);
	EXPECT_NEAR(mean_vector[24], 0.041654120309486, abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingSingleLaplacianWithLBFGS,get_variance_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethodWithLBFGS* inf
		= new CSingleLaplacianInferenceMethodWithLBFGS(kernel,
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

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare variance vector with result form GPML with the minfunc function
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	/*variance =
		0.999445535646085
		0.972965743674159
		0.799464093608188
		0.776811020003602
		0.957814443755535
		0.999871512927413
		0.982835632877678
		0.817449250977293
		0.721974547197594
		0.924548635855287
		0.996916479587550
		0.976888742629093
		0.969625640389031
		0.999882858745593
		0.994704144138483
		0.991865461515876
		0.916815204706781
		0.832493873837478
		0.920914793707155
		0.992188529246447
		0.998081902354648
		0.982979795460686
		0.970907685911889
		0.987229433547902
		0.998264934261243
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.999445535646085, rel_tolerance);
	EXPECT_NEAR(variance_vector[0],  0.999445535646085,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.972965743674159, rel_tolerance);
	EXPECT_NEAR(variance_vector[1],  0.972965743674159,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.799464093608188, rel_tolerance);
	EXPECT_NEAR(variance_vector[2],  0.799464093608188,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.776811020003602, rel_tolerance);
	EXPECT_NEAR(variance_vector[3],  0.776811020003602,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.957814443755535, rel_tolerance);
	EXPECT_NEAR(variance_vector[4],  0.957814443755535,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999871512927413, rel_tolerance);
	EXPECT_NEAR(variance_vector[5],  0.999871512927413,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.982835632877678, rel_tolerance);
	EXPECT_NEAR(variance_vector[6],  0.982835632877678,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.817449250977293, rel_tolerance);
	EXPECT_NEAR(variance_vector[7],  0.817449250977293,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.721974547197594, rel_tolerance);
	EXPECT_NEAR(variance_vector[8],  0.721974547197594,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.924548635855287, rel_tolerance);
	EXPECT_NEAR(variance_vector[9],  0.924548635855287,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996916479587550, rel_tolerance);
	EXPECT_NEAR(variance_vector[10], 0.996916479587550, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.976888742629093, rel_tolerance);
	EXPECT_NEAR(variance_vector[11], 0.976888742629093, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969625640389031, rel_tolerance);
	EXPECT_NEAR(variance_vector[12], 0.969625640389031, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999882858745593, rel_tolerance);
	EXPECT_NEAR(variance_vector[13], 0.999882858745593, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.994704144138483, rel_tolerance);
	EXPECT_NEAR(variance_vector[14], 0.994704144138483, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.991865461515876, rel_tolerance);
	EXPECT_NEAR(variance_vector[15], 0.991865461515876, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.916815204706781, rel_tolerance);
	EXPECT_NEAR(variance_vector[16], 0.916815204706781, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.832493873837478, rel_tolerance);
	EXPECT_NEAR(variance_vector[17], 0.832493873837478, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.920914793707155, rel_tolerance);
	EXPECT_NEAR(variance_vector[18], 0.920914793707155, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.992188529246447, rel_tolerance);
	EXPECT_NEAR(variance_vector[19], 0.992188529246447, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998081902354648, rel_tolerance);
	EXPECT_NEAR(variance_vector[20], 0.998081902354648, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.982979795460686, rel_tolerance);
	EXPECT_NEAR(variance_vector[21], 0.982979795460686, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.970907685911889, rel_tolerance);
	EXPECT_NEAR(variance_vector[22], 0.970907685911889, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.987229433547902, rel_tolerance);
	EXPECT_NEAR(variance_vector[23], 0.987229433547902, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998264934261243, rel_tolerance);
	EXPECT_NEAR(variance_vector[24], 0.998264934261243, abs_tolerance);


	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingSingleLaplacianWithLBFGS,get_probabilities)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
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

	// specify GP classification with SingleLaplacian inference
	CSingleLaplacianInferenceMethodWithLBFGS* inf=new CSingleLaplacianInferenceMethodWithLBFGS(kernel,
			features_train, mean, labels_train, likelihood);

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

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare probabilities with result form GPML with the minfunc function
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);

	/*probabilities =
		0.488226466245922
		0.417789511180478
		0.276093816652870
		0.263785590738910
		0.397304384814469
		0.494332392690781
		0.434493572227885
		0.286370205408434
		0.236359403085941
		0.362657942090414
		0.527764727350733
		0.576011934382649
		0.587141206106800
		0.505411594361785
		0.463613688406351
		0.545095837900176
		0.644208871583211
		0.704637561752594
		0.640610463004653
		0.544191265146420
		0.521898045707522
		0.565230752690983
		0.585282345866324
		0.556503466053284
		0.520827060710866
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.488226466245922, rel_tolerance);
	EXPECT_NEAR(probabilities[0],  0.488226466245922,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.417789511180478, rel_tolerance);
	EXPECT_NEAR(probabilities[1],  0.417789511180478,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.276093816652870, rel_tolerance);
	EXPECT_NEAR(probabilities[2],  0.276093816652870,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.263785590738910, rel_tolerance);
	EXPECT_NEAR(probabilities[3],  0.263785590738910,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.397304384814469, rel_tolerance);
	EXPECT_NEAR(probabilities[4],  0.397304384814469,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.494332392690781, rel_tolerance);
	EXPECT_NEAR(probabilities[5],  0.494332392690781,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.434493572227885, rel_tolerance);
	EXPECT_NEAR(probabilities[6],  0.434493572227885,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.286370205408434, rel_tolerance);
	EXPECT_NEAR(probabilities[7],  0.286370205408434,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.236359403085941, rel_tolerance);
	EXPECT_NEAR(probabilities[8],  0.236359403085941,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.362657942090414, rel_tolerance);
	EXPECT_NEAR(probabilities[9],  0.362657942090414,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.527764727350733, rel_tolerance);
	EXPECT_NEAR(probabilities[10], 0.527764727350733, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.576011934382649, rel_tolerance);
	EXPECT_NEAR(probabilities[11], 0.576011934382649, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.587141206106800, rel_tolerance);
	EXPECT_NEAR(probabilities[12], 0.587141206106800, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.505411594361785, rel_tolerance);
	EXPECT_NEAR(probabilities[13], 0.505411594361785, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.463613688406351, rel_tolerance);
	EXPECT_NEAR(probabilities[14], 0.463613688406351, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.545095837900176, rel_tolerance);
	EXPECT_NEAR(probabilities[15], 0.545095837900176, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.644208871583211, rel_tolerance);
	EXPECT_NEAR(probabilities[16], 0.644208871583211, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.704637561752594, rel_tolerance);
	EXPECT_NEAR(probabilities[17], 0.704637561752594, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.640610463004653, rel_tolerance);
	EXPECT_NEAR(probabilities[18], 0.640610463004653, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.544191265146420, rel_tolerance);
	EXPECT_NEAR(probabilities[19], 0.544191265146420, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.521898045707522, rel_tolerance);
	EXPECT_NEAR(probabilities[20], 0.521898045707522, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.565230752690983, rel_tolerance);
	EXPECT_NEAR(probabilities[21], 0.565230752690983, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.585282345866324, rel_tolerance);
	EXPECT_NEAR(probabilities[22], 0.585282345866324, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.556503466053284, rel_tolerance);
	EXPECT_NEAR(probabilities[23], 0.556503466053284, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.520827060710866, rel_tolerance);
	EXPECT_NEAR(probabilities[24], 0.520827060710866, abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingKLCovariance,get_mean_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCovarianceInferenceMethod* inf
	= new CKLCovarianceInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	/*mean =
	-0.016341488135989
	-0.115432982032852
	-0.310091528666790
	-0.327625480818566
	-0.141823379518906
	-0.006987928693087
	-0.087084390147747
	-0.281051978052473
	-0.343293847167883
	-0.174986161399922
	0.036291381149167
	0.092191006864351
	0.081263889121627
	-0.018557808448710
	-0.050375129345852
	0.057449116927526
	0.173627827604553
	0.232081013286083
	0.145428552473548
	0.040405229102009
	0.027946205212655
	0.080984402032326
	0.101041308158586
	0.061931242381067
	0.020693347644828
	*/

	abs_tolerance = CMath::get_abs_tolerance(-0.016341488135989, rel_tolerance);
	EXPECT_NEAR(mean_vector[0],  -0.016341488135989,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.115432982032852, rel_tolerance);
	EXPECT_NEAR(mean_vector[1],  -0.115432982032852,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.310091528666790, rel_tolerance);
	EXPECT_NEAR(mean_vector[2],  -0.310091528666790,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.327625480818566, rel_tolerance);
	EXPECT_NEAR(mean_vector[3],  -0.327625480818566,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.141823379518906, rel_tolerance);
	EXPECT_NEAR(mean_vector[4],  -0.141823379518906,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.006987928693087, rel_tolerance);
	EXPECT_NEAR(mean_vector[5],  -0.006987928693087,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.087084390147747, rel_tolerance);
	EXPECT_NEAR(mean_vector[6],  -0.087084390147747,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.281051978052473, rel_tolerance);
	EXPECT_NEAR(mean_vector[7],  -0.281051978052473,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.343293847167883, rel_tolerance);
	EXPECT_NEAR(mean_vector[8],  -0.343293847167883,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.174986161399922, rel_tolerance);
	EXPECT_NEAR(mean_vector[9],  -0.174986161399922,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.036291381149167, rel_tolerance);
	EXPECT_NEAR(mean_vector[10],  0.036291381149167,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.092191006864351, rel_tolerance);
	EXPECT_NEAR(mean_vector[11],  0.092191006864351,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.081263889121627, rel_tolerance);
	EXPECT_NEAR(mean_vector[12],  0.081263889121627,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.018557808448710, rel_tolerance);
	EXPECT_NEAR(mean_vector[13],  -0.018557808448710,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.050375129345852, rel_tolerance);
	EXPECT_NEAR(mean_vector[14],  -0.050375129345852,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.057449116927526, rel_tolerance);
	EXPECT_NEAR(mean_vector[15],  0.057449116927526,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.173627827604553, rel_tolerance);
	EXPECT_NEAR(mean_vector[16],  0.173627827604553,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.232081013286083, rel_tolerance);
	EXPECT_NEAR(mean_vector[17],  0.232081013286083,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.145428552473548, rel_tolerance);
	EXPECT_NEAR(mean_vector[18],  0.145428552473548,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.040405229102009, rel_tolerance);
	EXPECT_NEAR(mean_vector[19],  0.040405229102009,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.027946205212655, rel_tolerance);
	EXPECT_NEAR(mean_vector[20],  0.027946205212655,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.080984402032326, rel_tolerance);
	EXPECT_NEAR(mean_vector[21],  0.080984402032326,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.101041308158586, rel_tolerance);
	EXPECT_NEAR(mean_vector[22],  0.101041308158586,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061931242381067, rel_tolerance);
	EXPECT_NEAR(mean_vector[23],  0.061931242381067,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.020693347644828, rel_tolerance);
	EXPECT_NEAR(mean_vector[24],  0.020693347644828,  abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingKLCovariance, get_variance_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCovarianceInferenceMethod* inf
	= new CKLCovarianceInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	/*variance=
	0.999732955765501
	0.986675226659003
	0.903843243849093
	0.892661544318404
	0.979886129021836
	0.999951168852580
	0.992416308992595
	0.921009785632792
	0.882149334496674
	0.969379843318521
	0.998682935654286
	0.991500818253337
	0.993396180324828
	0.999655607745581
	0.997462346343389
	0.996699598964247
	0.969853377481323
	0.946138403272105
	0.978850536125449
	0.998367417461214
	0.999219009614212
	0.993441526627467
	0.989790654045602
	0.996164521217137
	0.999571785363250
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.999732955765501, rel_tolerance);
	EXPECT_NEAR(variance_vector[0],  0.999732955765501,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.986675226659003, rel_tolerance);
	EXPECT_NEAR(variance_vector[1],  0.986675226659003,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.903843243849093, rel_tolerance);
	EXPECT_NEAR(variance_vector[2],  0.903843243849093,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.892661544318404, rel_tolerance);
	EXPECT_NEAR(variance_vector[3],  0.892661544318404,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.979886129021836, rel_tolerance);
	EXPECT_NEAR(variance_vector[4],  0.979886129021836,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999951168852580, rel_tolerance);
	EXPECT_NEAR(variance_vector[5],  0.999951168852580,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.992416308992595, rel_tolerance);
	EXPECT_NEAR(variance_vector[6],  0.992416308992595,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.921009785632792, rel_tolerance);
	EXPECT_NEAR(variance_vector[7],  0.921009785632792,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.882149334496674, rel_tolerance);
	EXPECT_NEAR(variance_vector[8],  0.882149334496674,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969379843318521, rel_tolerance);
	EXPECT_NEAR(variance_vector[9],  0.969379843318521,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998682935654286, rel_tolerance);
	EXPECT_NEAR(variance_vector[10],  0.998682935654286,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.991500818253337, rel_tolerance);
	EXPECT_NEAR(variance_vector[11],  0.991500818253337,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.993396180324828, rel_tolerance);
	EXPECT_NEAR(variance_vector[12],  0.993396180324828,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999655607745581, rel_tolerance);
	EXPECT_NEAR(variance_vector[13],  0.999655607745581,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.997462346343389, rel_tolerance);
	EXPECT_NEAR(variance_vector[14],  0.997462346343389,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996699598964247, rel_tolerance);
	EXPECT_NEAR(variance_vector[15],  0.996699598964247,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969853377481323, rel_tolerance);
	EXPECT_NEAR(variance_vector[16],  0.969853377481323,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.946138403272105, rel_tolerance);
	EXPECT_NEAR(variance_vector[17],  0.946138403272105,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.978850536125449, rel_tolerance);
	EXPECT_NEAR(variance_vector[18],  0.978850536125449,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998367417461214, rel_tolerance);
	EXPECT_NEAR(variance_vector[19],  0.998367417461214,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999219009614212, rel_tolerance);
	EXPECT_NEAR(variance_vector[20],  0.999219009614212,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.993441526627467, rel_tolerance);
	EXPECT_NEAR(variance_vector[21],  0.993441526627467,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.989790654045602, rel_tolerance);
	EXPECT_NEAR(variance_vector[22],  0.989790654045602,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996164521217137, rel_tolerance);
	EXPECT_NEAR(variance_vector[23],  0.996164521217137,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999571785363250, rel_tolerance);
	EXPECT_NEAR(variance_vector[24],  0.999571785363250,  abs_tolerance);

	SG_UNREF(gpc);
	}


TEST(GaussianProcessClassificationUsingKLCovariance, get_probabilities)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCovarianceInferenceMethod* inf
	= new CKLCovarianceInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);
	/*probabilities=
	0.491829255932006
	0.442283508983574
	0.344954235666605
	0.336187259590717
	0.429088310240547
	0.496506035653456
	0.456457804926126
	0.359474010973764
	0.328353076416058
	0.412506919300039
	0.518145690574583
	0.546095503432175
	0.540631944560814
	0.490721095775645
	0.474812435327074
	0.528724558463763
	0.586813913802277
	0.616040506643041
	0.572714276236774
	0.520202614551004
	0.513973102606328
	0.540492201016163
	0.550520654079293
	0.530965621190533
	0.510346673822414
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.491829255932006, rel_tolerance);
	EXPECT_NEAR(probabilities[0],  0.491829255932006,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.442283508983574, rel_tolerance);
	EXPECT_NEAR(probabilities[1],  0.442283508983574,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.344954235666605, rel_tolerance);
	EXPECT_NEAR(probabilities[2],  0.344954235666605,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.336187259590717, rel_tolerance);
	EXPECT_NEAR(probabilities[3],  0.336187259590717,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.429088310240547, rel_tolerance);
	EXPECT_NEAR(probabilities[4],  0.429088310240547,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.496506035653456, rel_tolerance);
	EXPECT_NEAR(probabilities[5],  0.496506035653456,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.456457804926126, rel_tolerance);
	EXPECT_NEAR(probabilities[6],  0.456457804926126,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.359474010973764, rel_tolerance);
	EXPECT_NEAR(probabilities[7],  0.359474010973764,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.328353076416058, rel_tolerance);
	EXPECT_NEAR(probabilities[8],  0.328353076416058,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.412506919300039, rel_tolerance);
	EXPECT_NEAR(probabilities[9],  0.412506919300039,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.518145690574583, rel_tolerance);
	EXPECT_NEAR(probabilities[10],  0.518145690574583,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.546095503432175, rel_tolerance);
	EXPECT_NEAR(probabilities[11],  0.546095503432175,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.540631944560814, rel_tolerance);
	EXPECT_NEAR(probabilities[12],  0.540631944560814,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.490721095775645, rel_tolerance);
	EXPECT_NEAR(probabilities[13],  0.490721095775645,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.474812435327074, rel_tolerance);
	EXPECT_NEAR(probabilities[14],  0.474812435327074,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.528724558463763, rel_tolerance);
	EXPECT_NEAR(probabilities[15],  0.528724558463763,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.586813913802277, rel_tolerance);
	EXPECT_NEAR(probabilities[16],  0.586813913802277,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.616040506643041, rel_tolerance);
	EXPECT_NEAR(probabilities[17],  0.616040506643041,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.572714276236774, rel_tolerance);
	EXPECT_NEAR(probabilities[18],  0.572714276236774,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.520202614551004, rel_tolerance);
	EXPECT_NEAR(probabilities[19],  0.520202614551004,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.513973102606328, rel_tolerance);
	EXPECT_NEAR(probabilities[20],  0.513973102606328,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.540492201016163, rel_tolerance);
	EXPECT_NEAR(probabilities[21],  0.540492201016163,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.550520654079293, rel_tolerance);
	EXPECT_NEAR(probabilities[22],  0.550520654079293,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.530965621190533, rel_tolerance);
	EXPECT_NEAR(probabilities[23],  0.530965621190533,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.510346673822414, rel_tolerance);
	EXPECT_NEAR(probabilities[24],  0.510346673822414,  abs_tolerance);

	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLCholesky,get_mean_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCholeskyInferenceMethod* inf
	= new CKLCholeskyInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	/*mean =
	-0.016335300178810
	-0.115412365886247
	-0.310085405828793
	-0.327651270507445
	-0.141936552668286
	-0.006973283943821
	-0.087015591699728
	-0.281044745486262
	-0.343291280502099
	-0.175199940406874
	0.036264654510887
	0.092272078760991
	0.081310502536392
	-0.018547648424229
	-0.050537947293429
	0.057339417937389
	0.173526475014741
	0.232076894647369
	0.145465719897800
	0.040425760762898
	0.027867843552160
	0.080847982901803
	0.100974911211954
	0.061946134714323
	0.020730861749702
	*/

	abs_tolerance = CMath::get_abs_tolerance(-0.016335300178810, rel_tolerance);
	EXPECT_NEAR(mean_vector[0],  -0.016335300178810,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.115412365886247, rel_tolerance);
	EXPECT_NEAR(mean_vector[1],  -0.115412365886247,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.310085405828793, rel_tolerance);
	EXPECT_NEAR(mean_vector[2],  -0.310085405828793,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.327651270507445, rel_tolerance);
	EXPECT_NEAR(mean_vector[3],  -0.327651270507445,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.141936552668286, rel_tolerance);
	EXPECT_NEAR(mean_vector[4],  -0.141936552668286,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.006973283943821, rel_tolerance);
	EXPECT_NEAR(mean_vector[5],  -0.006973283943821,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.087015591699728, rel_tolerance);
	EXPECT_NEAR(mean_vector[6],  -0.087015591699728,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.281044745486262, rel_tolerance);
	EXPECT_NEAR(mean_vector[7],  -0.281044745486262,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.343291280502099, rel_tolerance);
	EXPECT_NEAR(mean_vector[8],  -0.343291280502099,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.175199940406874, rel_tolerance);
	EXPECT_NEAR(mean_vector[9],  -0.175199940406874,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.036264654510887, rel_tolerance);
	EXPECT_NEAR(mean_vector[10],  0.036264654510887,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.092272078760991, rel_tolerance);
	EXPECT_NEAR(mean_vector[11],  0.092272078760991,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.081310502536392, rel_tolerance);
	EXPECT_NEAR(mean_vector[12],  0.081310502536392,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.018547648424229, rel_tolerance);
	EXPECT_NEAR(mean_vector[13],  -0.018547648424229,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.050537947293429, rel_tolerance);
	EXPECT_NEAR(mean_vector[14],  -0.050537947293429,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.057339417937389, rel_tolerance);
	EXPECT_NEAR(mean_vector[15],  0.057339417937389,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.173526475014741, rel_tolerance);
	EXPECT_NEAR(mean_vector[16],  0.173526475014741,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.232076894647369, rel_tolerance);
	EXPECT_NEAR(mean_vector[17],  0.232076894647369,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.145465719897800, rel_tolerance);
	EXPECT_NEAR(mean_vector[18],  0.145465719897800,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.040425760762898, rel_tolerance);
	EXPECT_NEAR(mean_vector[19],  0.040425760762898,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.027867843552160, rel_tolerance);
	EXPECT_NEAR(mean_vector[20],  0.027867843552160,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.080847982901803, rel_tolerance);
	EXPECT_NEAR(mean_vector[21],  0.080847982901803,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.100974911211954, rel_tolerance);
	EXPECT_NEAR(mean_vector[22],  0.100974911211954,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061946134714323, rel_tolerance);
	EXPECT_NEAR(mean_vector[23],  0.061946134714323,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.020730861749702, rel_tolerance);
	EXPECT_NEAR(mean_vector[24],  0.020730861749702,  abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingKLCholesky, get_variance_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCovarianceInferenceMethod* inf
	= new CKLCovarianceInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	/*variance=
	0.999733157968068
	0.986679985800539
	0.903847041091993
	0.892644644934857
	0.979854015016643
	0.999951373311039
	0.992428286801146
	0.921013851034562
	0.882151096731230
	0.969304980881428
	0.998684874833206
	0.991485863481125
	0.993388602177279
	0.999655984737931
	0.997445915883367
	0.996712191150601
	0.969888562468958
	0.946140314970834
	0.978839724334615
	0.998365757866741
	0.999223383295752
	0.993463603660710
	0.989804067305738
	0.996162676393955
	0.999570231371115
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.999733157968068, rel_tolerance);
	EXPECT_NEAR(variance_vector[0],  0.999733157968068,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.986679985800539, rel_tolerance);
	EXPECT_NEAR(variance_vector[1],  0.986679985800539,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.903847041091993, rel_tolerance);
	EXPECT_NEAR(variance_vector[2],  0.903847041091993,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.892644644934857, rel_tolerance);
	EXPECT_NEAR(variance_vector[3],  0.892644644934857,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.979854015016643, rel_tolerance);
	EXPECT_NEAR(variance_vector[4],  0.979854015016643,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999951373311039, rel_tolerance);
	EXPECT_NEAR(variance_vector[5],  0.999951373311039,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.992428286801146, rel_tolerance);
	EXPECT_NEAR(variance_vector[6],  0.992428286801146,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.921013851034562, rel_tolerance);
	EXPECT_NEAR(variance_vector[7],  0.921013851034562,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.882151096731230, rel_tolerance);
	EXPECT_NEAR(variance_vector[8],  0.882151096731230,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969304980881428, rel_tolerance);
	EXPECT_NEAR(variance_vector[9],  0.969304980881428,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998684874833206, rel_tolerance);
	EXPECT_NEAR(variance_vector[10],  0.998684874833206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.991485863481125, rel_tolerance);
	EXPECT_NEAR(variance_vector[11],  0.991485863481125,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.993388602177279, rel_tolerance);
	EXPECT_NEAR(variance_vector[12],  0.993388602177279,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999655984737931, rel_tolerance);
	EXPECT_NEAR(variance_vector[13],  0.999655984737931,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.997445915883367, rel_tolerance);
	EXPECT_NEAR(variance_vector[14],  0.997445915883367,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996712191150601, rel_tolerance);
	EXPECT_NEAR(variance_vector[15],  0.996712191150601,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969888562468958, rel_tolerance);
	EXPECT_NEAR(variance_vector[16],  0.969888562468958,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.946140314970834, rel_tolerance);
	EXPECT_NEAR(variance_vector[17],  0.946140314970834,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.978839724334615, rel_tolerance);
	EXPECT_NEAR(variance_vector[18],  0.978839724334615,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998365757866741, rel_tolerance);
	EXPECT_NEAR(variance_vector[19],  0.998365757866741,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999223383295752, rel_tolerance);
	EXPECT_NEAR(variance_vector[20],  0.999223383295752,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.993463603660710, rel_tolerance);
	EXPECT_NEAR(variance_vector[21],  0.993463603660710,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.989804067305738, rel_tolerance);
	EXPECT_NEAR(variance_vector[22],  0.989804067305738,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996162676393955, rel_tolerance);
	EXPECT_NEAR(variance_vector[23],  0.996162676393955,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999570231371115, rel_tolerance);
	EXPECT_NEAR(variance_vector[24],  0.999570231371115,  abs_tolerance);

	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLCholesky, get_probabilities)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLCholeskyInferenceMethod* inf
	= new CKLCholeskyInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);
	/*probabilities=
	0.491832349910595
	0.442293817056876
	0.344957297085603
	0.336174364746277
	0.429031723665857
	0.496513358028089
	0.456492204150136
	0.359477627256869
	0.328354359748951
	0.412400029796563
	0.518132327255443
	0.546136039380496
	0.540655251268196
	0.490726175787885
	0.474731026353285
	0.528669708968695
	0.586763237507370
	0.616038447323685
	0.572732859948900
	0.520212880381449
	0.513933921776080
	0.540423991450902
	0.550487455605977
	0.530973067357162
	0.510365430874851
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.491832349910595, rel_tolerance);
	EXPECT_NEAR(probabilities[0],  0.491832349910595,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.442293817056876, rel_tolerance);
	EXPECT_NEAR(probabilities[1],  0.442293817056876,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.344957297085603, rel_tolerance);
	EXPECT_NEAR(probabilities[2],  0.344957297085603,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.336174364746277, rel_tolerance);
	EXPECT_NEAR(probabilities[3],  0.336174364746277,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.429031723665857, rel_tolerance);
	EXPECT_NEAR(probabilities[4],  0.429031723665857,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.496513358028089, rel_tolerance);
	EXPECT_NEAR(probabilities[5],  0.496513358028089,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.456492204150136, rel_tolerance);
	EXPECT_NEAR(probabilities[6],  0.456492204150136,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.359477627256869, rel_tolerance);
	EXPECT_NEAR(probabilities[7],  0.359477627256869,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.328354359748951, rel_tolerance);
	EXPECT_NEAR(probabilities[8],  0.328354359748951,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.412400029796563, rel_tolerance);
	EXPECT_NEAR(probabilities[9],  0.412400029796563,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.518132327255443, rel_tolerance);
	EXPECT_NEAR(probabilities[10],  0.518132327255443,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.546136039380496, rel_tolerance);
	EXPECT_NEAR(probabilities[11],  0.546136039380496,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.540655251268196, rel_tolerance);
	EXPECT_NEAR(probabilities[12],  0.540655251268196,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.490726175787885, rel_tolerance);
	EXPECT_NEAR(probabilities[13],  0.490726175787885,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.474731026353285, rel_tolerance);
	EXPECT_NEAR(probabilities[14],  0.474731026353285,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.528669708968695, rel_tolerance);
	EXPECT_NEAR(probabilities[15],  0.528669708968695,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.586763237507370, rel_tolerance);
	EXPECT_NEAR(probabilities[16],  0.586763237507370,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.616038447323685, rel_tolerance);
	EXPECT_NEAR(probabilities[17],  0.616038447323685,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.572732859948900, rel_tolerance);
	EXPECT_NEAR(probabilities[18],  0.572732859948900,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.520212880381449, rel_tolerance);
	EXPECT_NEAR(probabilities[19],  0.520212880381449,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.513933921776080, rel_tolerance);
	EXPECT_NEAR(probabilities[20],  0.513933921776080,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.540423991450902, rel_tolerance);
	EXPECT_NEAR(probabilities[21],  0.540423991450902,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.550487455605977, rel_tolerance);
	EXPECT_NEAR(probabilities[22],  0.550487455605977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.530973067357162, rel_tolerance);
	EXPECT_NEAR(probabilities[23],  0.530973067357162,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.510365430874851, rel_tolerance);
	EXPECT_NEAR(probabilities[24],  0.510365430874851,  abs_tolerance);
	
	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLApproxDiagonal,get_mean_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLApproxDiagonalInferenceMethod* inf
	= new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	/*mean =
	-0.015671229947670
	-0.113246817542353
	-0.330085372182225
	-0.324841589584311
	-0.132713568880395
	-0.006633296748899
	-0.085675868936265
	-0.297686201111398
	-0.366386103561268
	-0.173531826818146
	0.035385899994785
	0.094048289808513
	0.091906061664611
	-0.016486456808792
	-0.052971956478889
	0.054789408675094
	0.171735663002798
	0.247315776698091
	0.154249106884122
	0.042435572822795
	0.027023059600419
	0.076907492281476
	0.099616899680538
	0.061674380497058
	0.020610276974189
	*/

	abs_tolerance = CMath::get_abs_tolerance(-0.015671229947670, rel_tolerance);
	EXPECT_NEAR(mean_vector[0],  -0.015671229947670,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.113246817542353, rel_tolerance);
	EXPECT_NEAR(mean_vector[1],  -0.113246817542353,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.330085372182225, rel_tolerance);
	EXPECT_NEAR(mean_vector[2],  -0.330085372182225,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.324841589584311, rel_tolerance);
	EXPECT_NEAR(mean_vector[3],  -0.324841589584311,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.132713568880395, rel_tolerance);
	EXPECT_NEAR(mean_vector[4],  -0.132713568880395,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.006633296748899, rel_tolerance);
	EXPECT_NEAR(mean_vector[5],  -0.006633296748899,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.085675868936265, rel_tolerance);
	EXPECT_NEAR(mean_vector[6],  -0.085675868936265,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.297686201111398, rel_tolerance);
	EXPECT_NEAR(mean_vector[7],  -0.297686201111398,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.366386103561268, rel_tolerance);
	EXPECT_NEAR(mean_vector[8],  -0.366386103561268,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.173531826818146, rel_tolerance);
	EXPECT_NEAR(mean_vector[9],  -0.173531826818146,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.035385899994785, rel_tolerance);
	EXPECT_NEAR(mean_vector[10],  0.035385899994785,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.094048289808513, rel_tolerance);
	EXPECT_NEAR(mean_vector[11],  0.094048289808513,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.091906061664611, rel_tolerance);
	EXPECT_NEAR(mean_vector[12],  0.091906061664611,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.016486456808792, rel_tolerance);
	EXPECT_NEAR(mean_vector[13],  -0.016486456808792,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.052971956478889, rel_tolerance);
	EXPECT_NEAR(mean_vector[14],  -0.052971956478889,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.054789408675094, rel_tolerance);
	EXPECT_NEAR(mean_vector[15],  0.054789408675094,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.171735663002798, rel_tolerance);
	EXPECT_NEAR(mean_vector[16],  0.171735663002798,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.247315776698091, rel_tolerance);
	EXPECT_NEAR(mean_vector[17],  0.247315776698091,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.154249106884122, rel_tolerance);
	EXPECT_NEAR(mean_vector[18],  0.154249106884122,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.042435572822795, rel_tolerance);
	EXPECT_NEAR(mean_vector[19],  0.042435572822795,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.027023059600419, rel_tolerance);
	EXPECT_NEAR(mean_vector[20],  0.027023059600419,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.076907492281476, rel_tolerance);
	EXPECT_NEAR(mean_vector[21],  0.076907492281476,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.099616899680538, rel_tolerance);
	EXPECT_NEAR(mean_vector[22],  0.099616899680538,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061674380497058, rel_tolerance);
	EXPECT_NEAR(mean_vector[23],  0.061674380497058,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.020610276974189, rel_tolerance);
	EXPECT_NEAR(mean_vector[24],  0.020610276974189,  abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingKLApproxDiagonal, get_variance_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLApproxDiagonalInferenceMethod* inf
	= new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	/*variance=
	0.999754412551927
	0.987175158316529
	0.891043647071322
	0.894477941676338
	0.982387108635029
	0.999955999374241
	0.992659645482016
	0.911382925667864
	0.865761223117192
	0.969886705081157
	0.998747838081559
	0.991154919184094
	0.991553275829301
	0.999728196741892
	0.997193971826799
	0.996998120697034
	0.970506862052989
	0.938834906596220
	0.976207213025451
	0.998199222159201
	0.999269754249832
	0.994085237630975
	0.990076473298038
	0.996196270790304
	0.999575216483047
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.999754412551927, rel_tolerance);
	EXPECT_NEAR(variance_vector[0],  0.999754412551927,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.987175158316529, rel_tolerance);
	EXPECT_NEAR(variance_vector[1],  0.987175158316529,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.891043647071322, rel_tolerance);
	EXPECT_NEAR(variance_vector[2],  0.891043647071322,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.894477941676338, rel_tolerance);
	EXPECT_NEAR(variance_vector[3],  0.894477941676338,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.982387108635029, rel_tolerance);
	EXPECT_NEAR(variance_vector[4],  0.982387108635029,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999955999374241, rel_tolerance);
	EXPECT_NEAR(variance_vector[5],  0.999955999374241,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.992659645482016, rel_tolerance);
	EXPECT_NEAR(variance_vector[6],  0.992659645482016,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.911382925667864, rel_tolerance);
	EXPECT_NEAR(variance_vector[7],  0.911382925667864,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.865761223117192, rel_tolerance);
	EXPECT_NEAR(variance_vector[8],  0.865761223117192,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.969886705081157, rel_tolerance);
	EXPECT_NEAR(variance_vector[9],  0.969886705081157,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998747838081559, rel_tolerance);
	EXPECT_NEAR(variance_vector[10],  0.998747838081559,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.991154919184094, rel_tolerance);
	EXPECT_NEAR(variance_vector[11],  0.991154919184094,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.991553275829301, rel_tolerance);
	EXPECT_NEAR(variance_vector[12],  0.991553275829301,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999728196741892, rel_tolerance);
	EXPECT_NEAR(variance_vector[13],  0.999728196741892,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.997193971826799, rel_tolerance);
	EXPECT_NEAR(variance_vector[14],  0.997193971826799,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996998120697034, rel_tolerance);
	EXPECT_NEAR(variance_vector[15],  0.996998120697034,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.970506862052989, rel_tolerance);
	EXPECT_NEAR(variance_vector[16],  0.970506862052989,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.938834906596220, rel_tolerance);
	EXPECT_NEAR(variance_vector[17],  0.938834906596220,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.976207213025451, rel_tolerance);
	EXPECT_NEAR(variance_vector[18],  0.976207213025451,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998199222159201, rel_tolerance);
	EXPECT_NEAR(variance_vector[19],  0.998199222159201,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999269754249832, rel_tolerance);
	EXPECT_NEAR(variance_vector[20],  0.999269754249832,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.994085237630975, rel_tolerance);
	EXPECT_NEAR(variance_vector[21],  0.994085237630975,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.990076473298038, rel_tolerance);
	EXPECT_NEAR(variance_vector[22],  0.990076473298038,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.996196270790304, rel_tolerance);
	EXPECT_NEAR(variance_vector[23],  0.996196270790304,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999575216483047, rel_tolerance);
	EXPECT_NEAR(variance_vector[24],  0.999575216483047,  abs_tolerance);

	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLApproxDiagonal, get_probabilities)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitVGLikelihood* likelihood=new CLogitVGLikelihood();

	CKLApproxDiagonalInferenceMethod* inf
	= new CKLApproxDiagonalInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);
	/*probabilities=
	0.492164385026165
	0.443376591228824
	0.334957313908888
	0.337579205207844
	0.433643215559803
	0.496683351625550
	0.457162065531868
	0.351156899444301
	0.316806948219366
	0.413234086590927
	0.517692949997393
	0.547024144904257
	0.545953030832306
	0.491756771595604
	0.473514021760555
	0.527394704337547
	0.585867831501399
	0.623657888349045
	0.577124553442061
	0.521217786411397
	0.513511529800210
	0.538453746140738
	0.549808449840269
	0.530837190248529
	0.510305138487094
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.492164385026165, rel_tolerance);
	EXPECT_NEAR(probabilities[0],  0.492164385026165,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.443376591228824, rel_tolerance);
	EXPECT_NEAR(probabilities[1],  0.443376591228824,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.334957313908888, rel_tolerance);
	EXPECT_NEAR(probabilities[2],  0.334957313908888,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.337579205207844, rel_tolerance);
	EXPECT_NEAR(probabilities[3],  0.337579205207844,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.433643215559803, rel_tolerance);
	EXPECT_NEAR(probabilities[4],  0.433643215559803,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.496683351625550, rel_tolerance);
	EXPECT_NEAR(probabilities[5],  0.496683351625550,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.457162065531868, rel_tolerance);
	EXPECT_NEAR(probabilities[6],  0.457162065531868,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.351156899444301, rel_tolerance);
	EXPECT_NEAR(probabilities[7],  0.351156899444301,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.316806948219366, rel_tolerance);
	EXPECT_NEAR(probabilities[8],  0.316806948219366,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.413234086590927, rel_tolerance);
	EXPECT_NEAR(probabilities[9],  0.413234086590927,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.517692949997393, rel_tolerance);
	EXPECT_NEAR(probabilities[10],  0.517692949997393,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.547024144904257, rel_tolerance);
	EXPECT_NEAR(probabilities[11],  0.547024144904257,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.545953030832306, rel_tolerance);
	EXPECT_NEAR(probabilities[12],  0.545953030832306,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.491756771595604, rel_tolerance);
	EXPECT_NEAR(probabilities[13],  0.491756771595604,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.473514021760555, rel_tolerance);
	EXPECT_NEAR(probabilities[14],  0.473514021760555,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.527394704337547, rel_tolerance);
	EXPECT_NEAR(probabilities[15],  0.527394704337547,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.585867831501399, rel_tolerance);
	EXPECT_NEAR(probabilities[16],  0.585867831501399,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.623657888349045, rel_tolerance);
	EXPECT_NEAR(probabilities[17],  0.623657888349045,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.577124553442061, rel_tolerance);
	EXPECT_NEAR(probabilities[18],  0.577124553442061,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.521217786411397, rel_tolerance);
	EXPECT_NEAR(probabilities[19],  0.521217786411397,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.513511529800210, rel_tolerance);
	EXPECT_NEAR(probabilities[20],  0.513511529800210,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.538453746140738, rel_tolerance);
	EXPECT_NEAR(probabilities[21],  0.538453746140738,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.549808449840269, rel_tolerance);
	EXPECT_NEAR(probabilities[22],  0.549808449840269,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.530837190248529, rel_tolerance);
	EXPECT_NEAR(probabilities[23],  0.530837190248529,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.510305138487094, rel_tolerance);
	EXPECT_NEAR(probabilities[24],  0.510305138487094,  abs_tolerance);
	
	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLDual,get_mean_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	CKLDualInferenceMethod* inf
	= new CKLDualInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

	/*mean =
	-0.017636837591643
	-0.123720477772115
	-0.333807495648361
	-0.351774037881878
	-0.151401213227994
	-0.010363972910701
	-0.103267795616356
	-0.321230982497644
	-0.386913724158759
	-0.196404363818576
	0.028531994699231
	0.066152302604020
	0.033455531360725
	-0.070478010456282
	-0.077527927147277
	0.048505620395723
	0.150905551834610
	0.201840781193977
	0.118207632808578
	0.026241294489211
	0.023703081781903
	0.069673843680252
	0.088096466796064
	0.053864726312313
	0.017338490592413
	*/

	abs_tolerance = CMath::get_abs_tolerance(-0.017636837591643, rel_tolerance);
	EXPECT_NEAR(mean_vector[0],  -0.017636837591643,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.123720477772115, rel_tolerance);
	EXPECT_NEAR(mean_vector[1],  -0.123720477772115,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.333807495648361, rel_tolerance);
	EXPECT_NEAR(mean_vector[2],  -0.333807495648361,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.351774037881878, rel_tolerance);
	EXPECT_NEAR(mean_vector[3],  -0.351774037881878,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.151401213227994, rel_tolerance);
	EXPECT_NEAR(mean_vector[4],  -0.151401213227994,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.010363972910701, rel_tolerance);
	EXPECT_NEAR(mean_vector[5],  -0.010363972910701,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.103267795616356, rel_tolerance);
	EXPECT_NEAR(mean_vector[6],  -0.103267795616356,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.321230982497644, rel_tolerance);
	EXPECT_NEAR(mean_vector[7],  -0.321230982497644,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.386913724158759, rel_tolerance);
	EXPECT_NEAR(mean_vector[8],  -0.386913724158759,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.196404363818576, rel_tolerance);
	EXPECT_NEAR(mean_vector[9],  -0.196404363818576,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.028531994699231, rel_tolerance);
	EXPECT_NEAR(mean_vector[10],  0.028531994699231,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.066152302604020, rel_tolerance);
	EXPECT_NEAR(mean_vector[11],  0.066152302604020,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.033455531360725, rel_tolerance);
	EXPECT_NEAR(mean_vector[12],  0.033455531360725,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.070478010456282, rel_tolerance);
	EXPECT_NEAR(mean_vector[13],  -0.070478010456282,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.077527927147277, rel_tolerance);
	EXPECT_NEAR(mean_vector[14],  -0.077527927147277,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.048505620395723, rel_tolerance);
	EXPECT_NEAR(mean_vector[15],  0.048505620395723,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.150905551834610, rel_tolerance);
	EXPECT_NEAR(mean_vector[16],  0.150905551834610,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.201840781193977, rel_tolerance);
	EXPECT_NEAR(mean_vector[17],  0.201840781193977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.118207632808578, rel_tolerance);
	EXPECT_NEAR(mean_vector[18],  0.118207632808578,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.026241294489211, rel_tolerance);
	EXPECT_NEAR(mean_vector[19],  0.026241294489211,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.023703081781903, rel_tolerance);
	EXPECT_NEAR(mean_vector[20],  0.023703081781903,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.069673843680252, rel_tolerance);
	EXPECT_NEAR(mean_vector[21],  0.069673843680252,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.088096466796064, rel_tolerance);
	EXPECT_NEAR(mean_vector[22],  0.088096466796064,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.053864726312313, rel_tolerance);
	EXPECT_NEAR(mean_vector[23],  0.053864726312313,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.017338490592413, rel_tolerance);
	EXPECT_NEAR(mean_vector[24],  0.017338490592413,  abs_tolerance);

	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingKLDual, get_variance_vector)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	CKLDualInferenceMethod* inf
	= new CKLDualInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	//
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	/*variance=
	0.999688941959766
	0.984693243379839
	0.888572555848969
	0.876255026272279
	0.977077672633092
	0.999892588065506
	0.989335762388539
	0.896810655883598
	0.850297770057600
	0.961425325873020
	0.999185925278483
	0.995623872860186
	0.998880727421372
	0.995032850042124
	0.993989420512247
	0.997647204790026
	0.977227514425492
	0.959260299047005
	0.986026955545792
	0.999311394463530
	0.999438163914040
	0.995145555506820
	0.992239012538050
	0.997098591259300
	0.999699376743977
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.999688941959766, rel_tolerance);
	EXPECT_NEAR(variance_vector[0],  0.999688941959766,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.984693243379839, rel_tolerance);
	EXPECT_NEAR(variance_vector[1],  0.984693243379839,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.888572555848969, rel_tolerance);
	EXPECT_NEAR(variance_vector[2],  0.888572555848969,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.876255026272279, rel_tolerance);
	EXPECT_NEAR(variance_vector[3],  0.876255026272279,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.977077672633092, rel_tolerance);
	EXPECT_NEAR(variance_vector[4],  0.977077672633092,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999892588065506, rel_tolerance);
	EXPECT_NEAR(variance_vector[5],  0.999892588065506,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.989335762388539, rel_tolerance);
	EXPECT_NEAR(variance_vector[6],  0.989335762388539,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.896810655883598, rel_tolerance);
	EXPECT_NEAR(variance_vector[7],  0.896810655883598,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.850297770057600, rel_tolerance);
	EXPECT_NEAR(variance_vector[8],  0.850297770057600,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.961425325873020, rel_tolerance);
	EXPECT_NEAR(variance_vector[9],  0.961425325873020,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999185925278483, rel_tolerance);
	EXPECT_NEAR(variance_vector[10],  0.999185925278483,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.995623872860186, rel_tolerance);
	EXPECT_NEAR(variance_vector[11],  0.995623872860186,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.998880727421372, rel_tolerance);
	EXPECT_NEAR(variance_vector[12],  0.998880727421372,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.995032850042124, rel_tolerance);
	EXPECT_NEAR(variance_vector[13],  0.995032850042124,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.993989420512247, rel_tolerance);
	EXPECT_NEAR(variance_vector[14],  0.993989420512247,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.997647204790026, rel_tolerance);
	EXPECT_NEAR(variance_vector[15],  0.997647204790026,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.977227514425492, rel_tolerance);
	EXPECT_NEAR(variance_vector[16],  0.977227514425492,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.959260299047005, rel_tolerance);
	EXPECT_NEAR(variance_vector[17],  0.959260299047005,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.986026955545792, rel_tolerance);
	EXPECT_NEAR(variance_vector[18],  0.986026955545792,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999311394463530, rel_tolerance);
	EXPECT_NEAR(variance_vector[19],  0.999311394463530,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999438163914040, rel_tolerance);
	EXPECT_NEAR(variance_vector[20],  0.999438163914040,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.995145555506820, rel_tolerance);
	EXPECT_NEAR(variance_vector[21],  0.995145555506820,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.992239012538050, rel_tolerance);
	EXPECT_NEAR(variance_vector[22],  0.992239012538050,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.997098591259300, rel_tolerance);
	EXPECT_NEAR(variance_vector[23],  0.997098591259300,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.999699376743977, rel_tolerance);
	EXPECT_NEAR(variance_vector[24],  0.999699376743977,  abs_tolerance);

	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingKLDual, get_probabilities)
{
	float64_t abs_tolerance;
	float64_t rel_tolerance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

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
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	CKLDualInferenceMethod* inf
	= new CKLDualInferenceMethod(kernel,
		features_train,
		mean,
		labels_train,
		likelihood);

	// train Gaussian process binary classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form the following Matlab code with the minfunc function
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);
	/*probabilities=
	0.491181581204178
	0.438139761113942
	0.333096252175819
	0.324112981059061
	0.424299393386003
	0.494818013544649
	0.448366102191822
	0.339384508751178
	0.306543137920620
	0.401797818090712
	0.514265997349615
	0.533076151302010
	0.516727765680362
	0.464760994771859
	0.461236036426362
	0.524252810197862
	0.575452775917305
	0.600920390596989
	0.559103816404289
	0.513120647244606
	0.511851540890951
	0.534836921840126
	0.544048233398032
	0.526932363156156
	0.508669245296206
	*/

	abs_tolerance = CMath::get_abs_tolerance(0.491181581204178, rel_tolerance);
	EXPECT_NEAR(probabilities[0],  0.491181581204178,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.438139761113942, rel_tolerance);
	EXPECT_NEAR(probabilities[1],  0.438139761113942,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.333096252175819, rel_tolerance);
	EXPECT_NEAR(probabilities[2],  0.333096252175819,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.324112981059061, rel_tolerance);
	EXPECT_NEAR(probabilities[3],  0.324112981059061,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.424299393386003, rel_tolerance);
	EXPECT_NEAR(probabilities[4],  0.424299393386003,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.494818013544649, rel_tolerance);
	EXPECT_NEAR(probabilities[5],  0.494818013544649,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.448366102191822, rel_tolerance);
	EXPECT_NEAR(probabilities[6],  0.448366102191822,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.339384508751178, rel_tolerance);
	EXPECT_NEAR(probabilities[7],  0.339384508751178,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.306543137920620, rel_tolerance);
	EXPECT_NEAR(probabilities[8],  0.306543137920620,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.401797818090712, rel_tolerance);
	EXPECT_NEAR(probabilities[9],  0.401797818090712,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.514265997349615, rel_tolerance);
	EXPECT_NEAR(probabilities[10],  0.514265997349615,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.533076151302010, rel_tolerance);
	EXPECT_NEAR(probabilities[11],  0.533076151302010,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.516727765680362, rel_tolerance);
	EXPECT_NEAR(probabilities[12],  0.516727765680362,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.464760994771859, rel_tolerance);
	EXPECT_NEAR(probabilities[13],  0.464760994771859,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.461236036426362, rel_tolerance);
	EXPECT_NEAR(probabilities[14],  0.461236036426362,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.524252810197862, rel_tolerance);
	EXPECT_NEAR(probabilities[15],  0.524252810197862,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.575452775917305, rel_tolerance);
	EXPECT_NEAR(probabilities[16],  0.575452775917305,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.600920390596989, rel_tolerance);
	EXPECT_NEAR(probabilities[17],  0.600920390596989,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.559103816404289, rel_tolerance);
	EXPECT_NEAR(probabilities[18],  0.559103816404289,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.513120647244606, rel_tolerance);
	EXPECT_NEAR(probabilities[19],  0.513120647244606,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.511851540890951, rel_tolerance);
	EXPECT_NEAR(probabilities[20],  0.511851540890951,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.534836921840126, rel_tolerance);
	EXPECT_NEAR(probabilities[21],  0.534836921840126,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.544048233398032, rel_tolerance);
	EXPECT_NEAR(probabilities[22],  0.544048233398032,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.526932363156156, rel_tolerance);
	EXPECT_NEAR(probabilities[23],  0.526932363156156,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.508669245296206, rel_tolerance);
	EXPECT_NEAR(probabilities[24],  0.508669245296206,  abs_tolerance);
	
	SG_UNREF(gpc);
	}

TEST(GaussianProcessClassificationUsingMultiLaplacian,get_mean_vector)
{

	float64_t abs_tolerance;
	//the implementation used mc sampler
	//rel_tolerance is big
	float64_t rel_tolerance=1e-1;
	index_t n=10, m=10;
	const index_t C=2;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<index_t> lab_train(n);
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

	lab_train[0]=0;
	lab_train[1]=1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=1;
	lab_train[5]=0;
	lab_train[6]=1;
	lab_train[7]=0;
	lab_train[8]=0;
	lab_train[9]=1;

	feat_test(0, 0)=-2;
	feat_test(0, 1)=-2;
	feat_test(0, 2)=-2;
	feat_test(0, 3)=-2;
	feat_test(0, 4)=-2;
	feat_test(0, 5)=1;
	feat_test(0, 6)=1;
	feat_test(0, 7)=1;
	feat_test(0, 8)=1;
	feat_test(0, 9)=1;

	feat_test(1, 0)=-2;
	feat_test(1, 1)=-1;
	feat_test(1, 2)=0;
	feat_test(1, 3)=1;
	feat_test(1, 4)=2;
	feat_test(1, 5)=-2;
	feat_test(1, 6)=-1;
	feat_test(1, 7)=0;
	feat_test(1, 8)=1;
	feat_test(1, 9)=2;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CMulticlassLabels* labels_train=new CMulticlassLabels();
	labels_train->set_int_labels(lab_train);

	const float64_t ell=1.210875895826508;
	// choose Gaussian kernel with width = 2*ell^2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, ell*ell*2.0);
	CZeroMean* mean=new CZeroMean();

	CSoftMaxLikelihood* likelihood=new CSoftMaxLikelihood();
	CMultiLaplacianInferenceMethod* inf=new CMultiLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	const float64_t scale=CMath::sqrt(497.3965463400368);
	inf->set_scale(scale);

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare mean vector with result form GP-Stuff 4.4
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);
	SGMatrix<float64_t> mean_matrix(mean_vector.vector, C, m, false);

	//0.495018898849053   0.440144547864533   0.378181314460968  0.316866130884288  0.363482284945346  0.507788773680543 0.571331569524648  0.741011569366910   0.844409862561900   0.737462852233711
   //0.504981101150947   0.559855452135468   0.621818685539032  0.683133869115712  0.636517715054655  0.492211226319456 0.428668430475352  0.258988430633090   0.155590137438104   0.262537147766292

	abs_tolerance = CMath::get_abs_tolerance(0.495018898849053, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,0),  0.495018898849053,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.440144547864533, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,1),  0.440144547864533,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.378181314460968, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,2),  0.378181314460968,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.316866130884288, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,3),  0.316866130884288,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.363482284945346, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,4),  0.363482284945346,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.507788773680543, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,5),  0.507788773680543,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.571331569524648, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,6),  0.571331569524648,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.741011569366910, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,7),  0.741011569366910,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.844409862561900, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,8),  0.844409862561900,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.737462852233711, rel_tolerance);
	EXPECT_NEAR(mean_matrix(0,9),  0.737462852233711,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.504981101150947, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,0),  0.504981101150947,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.559855452135468, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,1),  0.559855452135468,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.621818685539032, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,2),  0.621818685539032,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.683133869115712, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,3),  0.683133869115712,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.636517715054655, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,4),  0.636517715054655,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.492211226319456, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,5),  0.492211226319456,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.428668430475352, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,6),  0.428668430475352,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.258988430633090, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,7),  0.258988430633090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.155590137438104, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,8),  0.155590137438104,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.262537147766292, rel_tolerance);
	EXPECT_NEAR(mean_matrix(1,9),  0.262537147766292,  abs_tolerance);
	
	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingMultiLaplacian,get_variance_vector)
{

	float64_t abs_tolerance;
	//the implementation used mc sampler
	//rel_tolerance is big
	float64_t rel_tolerance=1e-1;
	index_t n=10, m=10;
	const index_t C=2;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<index_t> lab_train(n);
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

	lab_train[0]=0;
	lab_train[1]=1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=1;
	lab_train[5]=0;
	lab_train[6]=1;
	lab_train[7]=0;
	lab_train[8]=0;
	lab_train[9]=1;

	feat_test(0, 0)=-2;
	feat_test(0, 1)=-2;
	feat_test(0, 2)=-2;
	feat_test(0, 3)=-2;
	feat_test(0, 4)=-2;
	feat_test(0, 5)=1;
	feat_test(0, 6)=1;
	feat_test(0, 7)=1;
	feat_test(0, 8)=1;
	feat_test(0, 9)=1;

	feat_test(1, 0)=-2;
	feat_test(1, 1)=-1;
	feat_test(1, 2)=0;
	feat_test(1, 3)=1;
	feat_test(1, 4)=2;
	feat_test(1, 5)=-2;
	feat_test(1, 6)=-1;
	feat_test(1, 7)=0;
	feat_test(1, 8)=1;
	feat_test(1, 9)=2;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CMulticlassLabels* labels_train=new CMulticlassLabels();
	labels_train->set_int_labels(lab_train);

	const float64_t ell=1.210875895826508;
	// choose Gaussian kernel with width = 2*ell^2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, ell*ell*2.0);
	CZeroMean* mean=new CZeroMean();

	CSoftMaxLikelihood* likelihood=new CSoftMaxLikelihood();
	CMultiLaplacianInferenceMethod* inf=new CMultiLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	const float64_t scale=CMath::sqrt(497.3965463400368);
	inf->set_scale(scale);

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// train gaussian process classifier
	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	// compare variance vector with result form the following Matlab code
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
	SGMatrix<float64_t> variance_matrix(variance_vector.vector, C, m, false);

	
	//0.249642303718200   0.246892076745162   0.237282230243870   0.216467221239879   0.230674540362865   0.249716846163020 0.243943184224802   0.188439563066926   0.131755968107814   0.197173723102424
	//0.249642303718200   0.246892076745162   0.237282230243870   0.216467221239878   0.230674540362865 0.249716846163020 0.243943184224802   0.188439563066926   0.131755968107815   0.197173723102423

	abs_tolerance = CMath::get_abs_tolerance(0.249642303718200, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,0),  0.249642303718200,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.246892076745162, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,1),  0.246892076745162,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.237282230243870, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,2),  0.237282230243870,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.216467221239879, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,3),  0.216467221239879,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.230674540362865, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,4),  0.230674540362865,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.249716846163020, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,5),  0.249716846163020,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.243943184224802, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,6),  0.243943184224802,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.188439563066926, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,7),  0.188439563066926,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.131755968107814, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,8),  0.131755968107814,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.197173723102424, rel_tolerance);
	EXPECT_NEAR(variance_matrix(0,9),  0.197173723102424,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.249642303718200, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,0),  0.249642303718200,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.246892076745162, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,1),  0.246892076745162,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.237282230243870, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,2),  0.237282230243870,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.216467221239878, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,3),  0.216467221239878,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.230674540362865, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,4),  0.230674540362865,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.249716846163020, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,5),  0.249716846163020,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.243943184224802, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,6),  0.243943184224802,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.188439563066926, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,7),  0.188439563066926,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.131755968107815, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,8),  0.131755968107815,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.197173723102423, rel_tolerance);
	EXPECT_NEAR(variance_matrix(1,9),  0.197173723102423,  abs_tolerance);
	
	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingMultiLaplacian,apply_multiclass)
{
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<index_t> lab_train(n);
	
	index_t m=3;
	SGMatrix<float64_t> feat_test(2, m);

	feat_train(0,0)=0.8822936;
	feat_train(0,1)=-0.7160792;
	feat_train(0,2)=0.9178174;
	feat_train(0,3)=-0.0135544;
	feat_train(0,4)=-0.5275911;

	feat_train(1,0)=-0.9597321;
	feat_train(1,1)=0.0231289;
	feat_train(1,2)=0.8284935;
	feat_train(1,3)=0.0023812;
	feat_train(1,4)=-0.7218931;

	lab_train[0]=0;
	lab_train[1]=1;
	lab_train[2]=0;
	lab_train[3]=2;
	lab_train[4]=1;

	feat_test(0,0)=0.8822936;
	feat_test(0,1)=-0.7160792;
	feat_test(0,2)=-0.0135544;

	feat_test(1,0)=-0.9597321;
	feat_test(1,1)=0.0231289;
	feat_test(1,2)=0.0023812;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CMulticlassLabels* labels_train=new CMulticlassLabels();
	labels_train->set_int_labels(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	const float64_t ell=0.829123236069650;
	CGaussianKernel* kernel=new CGaussianKernel(10, ell*ell*2.0);
	CZeroMean* mean=new CZeroMean();

	CSoftMaxLikelihood* likelihood=new CSoftMaxLikelihood();
	CMultiLaplacianInferenceMethod* inf=new CMultiLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	const float64_t scale=CMath::sqrt(5.114014937226176);
	inf->set_scale(scale);

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);
	gpc->train();

	CMulticlassLabels* prediction=gpc->apply_multiclass(features_test);
	SGVector<int32_t> p=prediction->get_int_labels();

	EXPECT_EQ(p[0], 0);
	EXPECT_EQ(p[1], 1);
	EXPECT_EQ(p[2], 2);

	SG_UNREF(gpc);
	SG_UNREF(prediction);
}


#ifdef HAVE_LINALG_LIB
TEST(GaussianProcessClassificationUsingSingleFITCLaplacian,get_mean_vector)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-2;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;

	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP regression with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise); 

	float64_t scale=4.0;
	inf->set_scale(scale);

	int32_t k=4;
	SGMatrix<float64_t> feat_test(dim, k);
	feat_test(0,0)=-0.81263;
	feat_test(0,1)=5.4576;
	feat_test(0,2)=-0.239;
	feat_test(0,3)=2.45;

	feat_test(1,0)=-0.5;
	feat_test(1,1)=0.69979;
	feat_test(1,2)=2.3546;
	feat_test(1,3)=-0.46;

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);

	// train model
	gpc->train();

	//result buggss
	SG_REF(features_test);
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);
	SG_UNREF(features_test);

	// compare variance vector with result form GPML 3.5
	abs_tolorance = CMath::get_abs_tolerance(0.489770829538461, rel_tolorance);
	EXPECT_NEAR(mean_vector[0],  0.489770829538461,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.321349595380404, rel_tolorance);
	EXPECT_NEAR(mean_vector[1],  0.321349595380404,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.403233721232556, rel_tolorance);
	EXPECT_NEAR(mean_vector[2],  -0.403233721232556,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.502096819177983, rel_tolorance);
	EXPECT_NEAR(mean_vector[3],  0.502096819177983,  abs_tolorance);

	// clean up
	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingSingleFITCLaplacian,get_variance_vector)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-2;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;

	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP regression with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise); 

	float64_t scale=4.0;
	inf->set_scale(scale);

	int32_t k=4;
	SGMatrix<float64_t> feat_test(dim, k);
	feat_test(0,0)=-0.81263;
	feat_test(0,1)=5.4576;
	feat_test(0,2)=-0.239;
	feat_test(0,3)=2.45;

	feat_test(1,0)=-0.5;
	feat_test(1,1)=0.69979;
	feat_test(1,2)=2.3546;
	feat_test(1,3)=-0.46;

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);

	// train model
	gpc->train();

	//result buggss
	SG_REF(features_test);
	SGVector<float64_t> var_vector=gpc->get_variance_vector(features_test);
	SG_UNREF(features_test);

	// compare variance vector with result form GPML 3.5
	abs_tolorance = CMath::get_abs_tolerance(0.760124534533208, rel_tolorance);
	EXPECT_NEAR(var_vector[0],  0.760124534533208,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.896734437548851, rel_tolorance);
	EXPECT_NEAR(var_vector[1],  0.896734437548851,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.837402566060945, rel_tolorance);
	EXPECT_NEAR(var_vector[2],  0.837402566060945,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.747898784171351, rel_tolorance);
	EXPECT_NEAR(var_vector[3],  0.747898784171351,  abs_tolorance);
	// clean up
	SG_UNREF(gpc);
}

TEST(GaussianProcessClassificationUsingSingleFITCLaplacian,get_probabilities)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-2;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;

	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP regression with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise); 

	float64_t scale=4.0;
	inf->set_scale(scale);

	int32_t k=4;
	SGMatrix<float64_t> feat_test(dim, k);
	feat_test(0,0)=-0.81263;
	feat_test(0,1)=5.4576;
	feat_test(0,2)=-0.239;
	feat_test(0,3)=2.45;

	feat_test(1,0)=-0.5;
	feat_test(1,1)=0.69979;
	feat_test(1,2)=2.3546;
	feat_test(1,3)=-0.46;

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessClassification* gpc=new CGaussianProcessClassification(inf);

	// train model
	gpc->train();

	SG_REF(features_test);
	SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);
	SG_UNREF(features_test);

	// compare variance vector with result form GPML 3.5
	abs_tolorance = CMath::get_abs_tolerance(0.744885414769230, rel_tolorance);
	EXPECT_NEAR(probabilities[0],  0.744885414769230,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.660674797690202, rel_tolorance);
	EXPECT_NEAR(probabilities[1],  0.660674797690202,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.298383139383722, rel_tolorance);
	EXPECT_NEAR(probabilities[2],  0.298383139383722,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.751048409588992, rel_tolorance);
	EXPECT_NEAR(probabilities[3],  0.751048409588992,  abs_tolorance);

	// clean up
	SG_UNREF(gpc);
}

#endif /* HAVE_LINALG_LIB */


#endif /* HAVE_EIGEN3 */
