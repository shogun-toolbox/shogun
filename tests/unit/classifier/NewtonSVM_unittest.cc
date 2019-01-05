/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */
#include <gtest/gtest.h>
 
#include <functional>
#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>

#include "environments/LinearTestEnvironment.h"
#include <shogun/base/some.h>
#include <shogun/classifier/svm/NewtonSVM.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

extern LinearTestEnvironment* linear_test_env;

TEST(NewtonSVM, continue_training_consistency)
{
	auto env = linear_test_env->getBinaryLabelData();
	auto features = wrap(env->get_features_train());
	auto labels = wrap(env->get_labels_train());
	auto test_features = wrap(env->get_features_test());
	auto test_labels = wrap(env->get_labels_test());

	auto svm = some<CNewtonSVM>();
	svm->set_labels(labels);
	svm->train(features);

	// reference for completly trained model
	auto results = svm->apply(test_features);

	index_t iter = 0;

	// prematurely stopped at 5 iterations
	auto svm_stop = some<CNewtonSVM>();

	std::function<bool()> callback = [&iter]() {
		if (iter >= 5)
		{
			get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
			return true;
		}
		iter++;
		return false;
	};
	svm_stop->set_callback(callback);

	svm_stop->set_labels(labels);
	svm_stop->train(features);

	// callback executes, model should not be converged
	ASSERT(!svm_stop->is_complete());

	// reference model for intermediate state
	auto svm_one = some<CNewtonSVM>();
	svm_one->set_labels(labels);

	// trained only till 5 iterations
	svm_one->put<int32_t>("max_iterations", 5);
	svm_one->train(features);

	auto results_one = svm_one->apply(test_features);
	auto results_stop = svm_stop->apply(test_features);

	// prematurely stopped model and intermediate reference model should be
	// consistent
	EXPECT_TRUE(results_one->equals(results_stop));

	svm_stop->set_callback(nullptr);

	// continue training until converged
	svm_stop->continue_train();
	auto results_complete = svm_stop->apply(test_features);

	// compare model with completely trained reference
	EXPECT_TRUE(results_complete->equals(results));

	SG_UNREF(results_one);
	SG_UNREF(results_stop);
	SG_UNREF(results);
	SG_UNREF(results_complete);
}
