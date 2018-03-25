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
	CMath::init_random(5);
	std::shared_ptr<GaussianCheckerboard> mockData =
	    linear_test_env->getBinaryLabelData();

	CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
	CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();

	CBinaryLabels* ground_truth = (CBinaryLabels*)mockData->get_labels_test();

	CSVMOcas* ocas = new CSVMOcas(1.0, train_feats, ground_truth);
	ocas->parallel->set_num_threads(1);
	ocas->set_epsilon(1e-5);
	ocas->train();
	float64_t objective = ocas->compute_primal_objective();

	EXPECT_NEAR(objective, 0.024344632618686062, 1e-2);

	CLabels* pred = ocas->apply(test_feats);
	SG_REF(pred);
	CAccuracyMeasure evaluate = CAccuracyMeasure();
	evaluate.evaluate(pred, ground_truth);
	EXPECT_GT(evaluate.get_accuracy(), 0.99);

	SG_UNREF(ocas);
	SG_UNREF(pred);
}
#endif // HAVE_LAPACK
