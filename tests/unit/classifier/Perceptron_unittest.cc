/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2017 Michele Mazzoni
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are
* those
* of the authors and should not be interpreted as representing official
* policies,
* either expressed or implied, of the Shogun Development Team.
*/
#include <functional>
#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>

#include "environments/LinearTestEnvironment.h"
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

extern LinearTestEnvironment* linear_test_env;

TEST(Perceptron, train)
{
	auto env = linear_test_env->getBinaryLabelData();
	auto features = wrap(env->get_features_train());
	auto labels = wrap(env->get_labels_train());
	auto test_features = wrap(env->get_features_test());
	auto test_labels = wrap(env->get_labels_test());

	auto perceptron = some<CPerceptron>();
	perceptron->set_labels(labels);
	EXPECT_TRUE(perceptron->train(features));

	auto results = wrap(perceptron->apply(test_features));
	auto acc = some<CAccuracyMeasure>();
	EXPECT_EQ(acc->evaluate(results, test_labels), 1.0);
}

TEST(Perceptron, custom_hyperplane_initialization)
{
	auto env = linear_test_env->getBinaryLabelData();
	auto features = wrap(env->get_features_train());
	auto labels = wrap(env->get_labels_train());
	auto test_features = wrap(env->get_features_test());
	auto test_labels = wrap(env->get_labels_test());

	auto perceptron = some<CPerceptron>();
	perceptron->set_labels(labels);
	perceptron->train(features);

	auto weights = perceptron->get_w();

	auto perceptron_initialized = some<CPerceptron>();
	perceptron_initialized->set_initialize_hyperplane(false);
	perceptron_initialized->set_w(weights);
	perceptron_initialized->put<int32_t>("max_iterations", 1);
	perceptron_initialized->set_labels(labels);

	perceptron_initialized->train(features);
	EXPECT_TRUE(perceptron_initialized->get_w().equals(weights));
}

TEST(Perceptron, continue_training_consistency)
{
	auto env = linear_test_env->getBinaryLabelData();
	auto features = wrap(env->get_features_train());
	auto labels = wrap(env->get_labels_train());
	auto test_features = wrap(env->get_features_test());
	auto test_labels = wrap(env->get_labels_test());

	auto perceptron = some<CPerceptron>();
	perceptron->set_labels(labels);
	perceptron->train(features);

	auto results = perceptron->apply_binary(test_features);

	index_t iter = 0;

	auto perceptron_stop = some<CPerceptron>();

	std::function<bool()> callback = [&iter]() {
		if (iter >= 1)
		{
			get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
			return true;
		}
		iter++;
		return false;
	};
	perceptron_stop->set_callback(callback);

	perceptron_stop->put<bool>("compute_bias", false);
	perceptron_stop->set_labels(labels);
	perceptron_stop->train(features);

	EXPECT_FALSE(perceptron_stop->get<bool>("complete"));

	if (!perceptron_stop->get<bool>("complete"))
	{
		perceptron_stop->set_callback(nullptr);
		perceptron_stop->continue_train();
	}

	auto results_complete = perceptron_stop->apply_binary(test_features);

	EXPECT_TRUE(results_complete->get_labels().equals(results->get_labels()));

	SG_UNREF(results);
	SG_UNREF(results_complete);
}
