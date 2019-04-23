/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Evgeniy Andreev, Fernando Iglesias
 */

#include <shogun/base/init.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <shogun/evaluation/MeanSquaredError.h>

using namespace shogun;

void test_libsvr()
{
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
		feat_train[i]=Math::random(0.0, x_range);
		feat_test[i]=(float64_t)i/n*x_range;
		lab_train[i] = std::sin(feat_train[i]);
		lab_test[i] = std::sin(feat_test[i]);
	}

	/* shogun representation */
	auto labels_train=std::make_shared<RegressionLabels>(lab_train);
	auto labels_test=std::make_shared<RegressionLabels>(lab_test);
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(
			feat_train);
	auto features_test=std::make_shared<DenseFeatures<float64_t>>(
			feat_test);

	auto kernel=std::make_shared<GaussianKernel>(rbf_width);
	kernel->init(features_train, features_train);

	// also epsilon svr possible here
	LIBSVR_SOLVER_TYPE st=LIBSVR_NU_SVR;
	auto svm=std::make_shared<LibSVR>(svm_C, svm_nu, kernel, labels_train, st);
	svm->train();

	/* predict */
	auto predicted_labels =
	    svm->apply(features_test)->as<RegressionLabels>();

	/* evaluate */
	auto eval=std::make_shared<MeanSquaredError>();
	SG_SPRINT("mean squared error: %f\n",
			eval->evaluate(predicted_labels, labels_test));

	 /* clean up */
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_libsvr();

	exit_shogun();
	return 0;
}

