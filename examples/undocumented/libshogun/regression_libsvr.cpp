/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */
#include <kernel/GaussianKernel.h>
#include <labels/RegressionLabels.h>
#include <features/DenseFeatures.h>
#include <regression/svr/LibSVR.h>
#include <evaluation/MeanSquaredError.h>


using namespace shogun;

void test_libsvr()
{
	const int32_t kernel_cache=0;
	const float64_t rbf_width=10;
	const float64_t svm_C=10;
	const float64_t svm_nu=0.01;

	/* create some easy regression data: 1d noisy sine wave */
	index_t n=100;
	float64_t x_range=6;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> feat_test(1, n);
	SGVector<float64_t> lab_train(n);
	SGVector<float64_t> lab_test(n);

	for (index_t i=0; i<n; ++i)
	{
		feat_train[i]=CMath::random(0.0, x_range);
		feat_test[i]=(float64_t)i/n*x_range;
		lab_train[i]=CMath::sin(feat_train[i]);
		lab_test[i]=CMath::sin(feat_test[i]);
	}

	/* shogun representation */
	CLabels* labels_train=new CRegressionLabels(lab_train);
	CLabels* labels_test=new CRegressionLabels(lab_test);
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
			feat_test);

	CGaussianKernel* kernel=new CGaussianKernel(kernel_cache, rbf_width);
	kernel->init(features_train, features_train);

	// also epsilon svr possible here
	LIBSVR_SOLVER_TYPE st=LIBSVR_NU_SVR;
	CLibSVR* svm=new CLibSVR(svm_C, svm_nu, kernel, labels_train, st);
	svm->train();

	/* predict */
	CRegressionLabels* predicted_labels=CLabelsFactory::to_regression(
			svm->apply(features_test));

	/* evaluate */
	CEvaluation* eval=new CMeanSquaredError();
	SG_SPRINT("mean squared error: %f\n",
			eval->evaluate(predicted_labels, labels_test));

	 /* clean up */
	SG_UNREF(eval);
	SG_UNREF(labels_test)
	SG_UNREF(predicted_labels);
	SG_UNREF(svm);
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_libsvr();

	exit_shogun();
	return 0;
}

