/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#ifdef HAVE_EIGEN3
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#endif
#include <shogun/io/SerializableAsciiFile.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGObject,equals_null)
{
	CBinaryLabels* labels=new CBinaryLabels(10);

	EXPECT_FALSE(labels->equals(NULL));

	SG_UNREF(labels);
}

TEST(SGObject,equals_different_name)
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	CRegressionLabels* labels2=new CRegressionLabels(10);

	EXPECT_FALSE(labels->equals(labels2));

	SG_UNREF(labels);
	SG_UNREF(labels2);
}

#ifdef HAVE_EIGEN3
TEST(SGObject,equals_complex_equal)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=100;
	float64_t x_range=6;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	for (index_t  i=0; i<n; ++i)
	{
		X[i]=CMath::random(0.0, x_range);
		X_test[i]=(float64_t)i / n*x_range;
		Y[i]=CMath::sin(X[i]);
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CFeatures* latent=inf->get_latent_features();
	SG_UNREF(latent);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf,
			feat_train, label_train);

	/* perform inference */
	gpr->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);
	predictions->get_labels().display_vector("predictions");

	/* save and load instance to compare */
	const char* filename_gpr="gpr_instance.txt";
	const char* filename_predictions="predictions_instance.txt";

	CSerializableAsciiFile* file;
	file=new CSerializableAsciiFile(filename_gpr, 'w');
	gpr->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_predictions, 'w');
	predictions->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_gpr, 'r');
	CGaussianProcessRegression* gpr_copy=new CGaussianProcessRegression();
	gpr_copy->load_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename_predictions, 'r');
	CRegressionLabels* predictions_copy=new CRegressionLabels();
	predictions_copy->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* now compare */
	floatmax_t accuracy=1E-15;
	EXPECT_TRUE(predictions->equals(predictions_copy, accuracy));
	EXPECT_TRUE(gpr->equals(gpr_copy, accuracy));

	SG_UNREF(predictions);
	SG_UNREF(predictions_copy);
	SG_UNREF(gpr);
	SG_UNREF(gpr_copy);
}
#endif //HAVE_EIGEN3
