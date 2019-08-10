#include <gtest/gtest.h>

#include "environments/GaussianCheckerboard.h"
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/config.h>

using namespace shogun;

#ifdef USE_SVMLIGHT
TEST(SVMLight, CacheSizeInvariance)
{
	const int32_t num_samples = 50;
	const int32_t num_labels = 2;
	const int32_t num_dims = 5;
	const int32_t seed = 125;
	const float64_t epsilon = 1e-5;
	const float64_t svm_c = 1.0;
	const float64_t kernel_width = 1.2;
	const auto cache_sizes = {0, 1, 2, 3};

	std::mt19937_64 prng(seed);
	auto mockData = std::make_shared<GaussianCheckerboard>(
	    num_samples, num_labels, num_dims, prng);

	auto train_feats = mockData->get_features_train();
	auto train_labels = mockData->get_labels_train();
	auto test_feats = mockData->get_features_test();

	auto reference_labels = wrap<CBinaryLabels>(nullptr);
	for (auto cache_size : cache_sizes)
	{
		auto gaussian_kernel =
		    some<CGaussianKernel>(train_feats, train_feats, kernel_width);
		gaussian_kernel->set_cache_size(cache_size);

		auto svmlight = some<CSVMLight>(svm_c, gaussian_kernel, train_labels);
		svmlight->set_epsilon(epsilon);
		svmlight->train();

		auto labels_predict =
		    wrap(svmlight->apply(test_feats)->as<CBinaryLabels>());

		if (reference_labels.get() == nullptr)
			reference_labels = labels_predict;
		else
			EXPECT_TRUE(labels_predict->get_labels().equals(
			    reference_labels->get_labels()));
	}
}
#endif // USE_SVMLIGHT
