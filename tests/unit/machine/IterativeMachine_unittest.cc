/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 */

#include <gtest/gtest.h>

#include <functional>
#include <rxcpp/rx-lite.hpp>
#include <shogun/lib/Signal.h>

#include "environments/LinearTestEnvironment.h"
#include "utils/SGObjectIterator.h"
#include <shogun/classifier/AveragedPerceptron.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/classifier/svm/NewtonSVM.h>
#include <shogun/machine/IterativeMachine.h>
#include <shogun/machine/LinearMachine.h>

using namespace shogun;

std::set<std::string> sg_linear_machines = {"Perceptron", "AveragedPerceptron",
                                            "NewtonSVM"};
extern LinearTestEnvironment* linear_test_env;

//fixme
#if 0
TEST(IterativeMachine, continue_training_consistency)
{
	auto env = linear_test_env->getBinaryLabelData();
	auto features = wrap(env->get_features_train());
	auto labels = wrap(env->get_labels_train());
	auto test_features = wrap(env->get_features_test());
	auto test_labels = wrap(env->get_labels_test());

	auto range = sg_object_iterator<untemplated_sgobject>(sg_linear_machines);
	for (auto machine_obj : range)
	{
		// to know if a test fails, which machine is the culprit
		SCOPED_TRACE(machine_obj->get_name());

		auto machine = machine_obj->->as<IterativeMachine>();
		auto machine_stop =
		    machine->clone()->as<IterativeMachine<LinearMachine>>();
		auto machine_iters =
		    machine->clone()->as<IterativeMachine<LinearMachine>>();

		// get result of fully trained machine
		machine->set_labels(labels);
		machine->train(features);
		auto results = machine->apply(test_features);

		const int32_t max_iters = 2;
		index_t iter = 0;
		auto callback = [&iter, max_iters]() {
			if (iter >= max_iters)
			{
				get_global_signal()->get_subscriber()->on_next(SG_BLOCK_COMP);
				return true;
			}
			iter++;
			return false;
		};
		machine_stop->set_callback(callback);
		machine_stop->set_labels(labels);
		machine_stop->train(features);

		machine_iters->set_labels(labels);
		machine_iters->put<int32_t>("max_iterations", max_iters);
		machine_iters->train(features);

		auto results_iters = machine_iters->apply(test_features);
		auto results_stop = machine_stop->apply(test_features);
		EXPECT_TRUE(results_iters->equals(results_stop));

		machine_stop->set_callback(nullptr);
		machine_stop->continue_train();

		auto results_complete = machine_stop->apply(test_features);
		EXPECT_TRUE(results_complete->equals(results));

	}
}
#endif