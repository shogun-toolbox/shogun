#include <gtest/gtest.h>

#include "environments/MultiLabelTestEnvironment.h"
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/MulticlassOCAS.h>

using namespace shogun;

extern MultiLabelTestEnvironment* multilabel_test_env;

#ifdef HAVE_LAPACK
TEST(MulticlassOCASTest,train)
{
  float64_t C = 1.0;
  std::shared_ptr<GaussianCheckerboard> mockData =
	  multilabel_test_env->getMulticlassFixture();

  auto train_feats = mockData->get_features_train();
  auto test_feats = mockData->get_features_test();
  auto ground_truth =
	  std::static_pointer_cast<MulticlassLabels>(mockData->get_labels_test());
  auto mocas = std::make_shared<MulticlassOCAS>(C, train_feats, ground_truth);
  env()->set_num_threads(1);
  mocas->set_epsilon(1e-5);
  mocas->train();

  auto pred = mocas->apply(test_feats)->as<MulticlassLabels>();

  MulticlassAccuracy evaluate;
  float64_t result = evaluate.evaluate(pred, ground_truth);
  EXPECT_GT(result, 0.99);
}
#endif // HAVE_LAPACK
