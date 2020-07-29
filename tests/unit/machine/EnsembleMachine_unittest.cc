
#include "environments/MultiLabelTestEnvironment.h"
#include <gtest/gtest.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/Composite.h>
#include <shogun/machine/EnsembleMachine.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/MulticlassOCAS.h>

using namespace shogun;
extern MultiLabelTestEnvironment* multilabel_test_env;

TEST(EnsembleMachine, train)
{
	std::shared_ptr<GaussianCheckerboard> mockData =
	    multilabel_test_env->getMulticlassFixture();

	auto train_feats = mockData->get_features_train();
	auto test_feats = mockData->get_features_test();
	auto train_labels = mockData->get_labels_train();
	auto ground_truth =
	    std::static_pointer_cast<MulticlassLabels>(mockData->get_labels_test());

	std::vector<std::shared_ptr<Machine>> lists{
		std::make_shared<MulticlassLibLinear>(),
	    std::make_shared<MulticlassOCAS>()
	};
	auto ensemble = std::make_shared<EnsembleMachine>(lists);
	ensemble->set_combination_rule(std::make_shared<MeanRule>());
	ensemble->train(train_feats, train_labels);

	auto pred = ensemble->apply_multiclass(test_feats)->as<MulticlassLabels>();
	MulticlassAccuracy evaluate;
	float64_t result = evaluate.evaluate(pred, ground_truth);
	EXPECT_NEAR(result, 1.0,  std::numeric_limits<float64_t>::epsilon());
}

TEST(Composite, train)
{
	std::shared_ptr<GaussianCheckerboard> mockData =
	    multilabel_test_env->getMulticlassFixture();

	auto train_feats = mockData->get_features_train();
	auto test_feats = mockData->get_features_test();
	auto train_labels = mockData->get_labels_train();
	auto ground_truth =
	    std::static_pointer_cast<MulticlassLabels>(mockData->get_labels_test());

	auto composite = std::make_shared<Composite>();
	auto pred = composite->with(std::make_shared<MulticlassLibLinear>())
	                ->with(std::make_shared<MulticlassOCAS>())
	                ->then(std::make_shared<MeanRule>())
	                ->train(train_feats, train_labels)
	                ->apply_multiclass(test_feats);

	MulticlassAccuracy evaluate;
	float64_t result = evaluate.evaluate(pred, ground_truth);
	EXPECT_NEAR(result, 1.0, std::numeric_limits<float64_t>::epsilon());
}