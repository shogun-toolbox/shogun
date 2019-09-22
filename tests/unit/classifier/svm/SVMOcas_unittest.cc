#include <gtest/gtest.h>

#include <shogun/lib/config.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>

#include "environments/LinearTestEnvironment.h"

using namespace shogun;

extern LinearTestEnvironment* linear_test_env;

#ifdef HAVE_LAPACK
TEST(SVMOcasTest,train)
{
	std::shared_ptr<GaussianCheckerboard> mockData =
	    linear_test_env->getBinaryLabelData();

	auto train_feats = mockData->get_features_train();
	auto test_feats = mockData->get_features_test();

	auto ground_truth = std::static_pointer_cast<BinaryLabels>(mockData->get_labels_test());

	auto ocas = std::make_shared<SVMOcas>(1.0, train_feats, ground_truth);
	env()->set_num_threads(1);
	ocas->set_epsilon(1e-5);
	ocas->train();
	float64_t objective = ocas->compute_primal_objective();

	EXPECT_NEAR(objective, 0.024344632618686062, 1e-2);

	auto pred = ocas->apply(test_feats);

	auto evaluate = std::make_shared<AccuracyMeasure>();
	evaluate->evaluate(pred, ground_truth);
	EXPECT_GT(evaluate->get_accuracy(), 0.99);

}
#endif // HAVE_LAPACK
