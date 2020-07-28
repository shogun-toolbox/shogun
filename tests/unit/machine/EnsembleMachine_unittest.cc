
#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/EnsembleMachine.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/MulticlassOCAS.h>
using namespace shogun;

TEST(EnsembleMachine, train)
{
	int32_t seed = 100;
	index_t num_vec = 10;
	index_t num_feat = 3;
	index_t num_class = num_feat; // to make data easy
	float64_t distance = 15;

	// create some linearly seperable data
	SGMatrix<float64_t> matrix(num_class, num_vec);
	SGMatrix<float64_t> matrix_test(num_class, num_vec);
	auto labels = std::make_shared<MulticlassLabels>(num_vec);
	auto labels_test = std::make_shared<MulticlassLabels>(num_vec);
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	for (index_t i = 0; i < num_vec; ++i)
	{
		index_t label = i % num_class;
		for (index_t j = 0; j < num_feat; ++j)
		{
			matrix(j, i) = normal_dist(prng);
			matrix_test(j, i) = normal_dist(prng);
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label, i) += distance;
		matrix_test(label, i) += distance;
	}
	auto features = std::make_shared<DenseFeatures<float64_t>>(matrix);
	auto features_test =
	    std::make_shared<DenseFeatures<float64_t>>(matrix_test);

	std::vector<std::shared_ptr<Machine>> lists{
	    std::make_shared<MulticlassLibLinear>(),
	    std::make_shared<MulticlassOCAS>()};
	auto ensemble = std::make_shared<EnsembleMachine>(lists);
	ensemble->set_combination_rule(std::make_shared<MajorityVote>());
	ensemble->train(features, labels);

	auto pred =
	    ensemble->apply_multiclass(features_test)->as<MulticlassLabels>();
	MulticlassAccuracy evaluate;
	float64_t result = evaluate.evaluate(pred, labels_test);
	EXPECT_GE(result, 0.4);
}