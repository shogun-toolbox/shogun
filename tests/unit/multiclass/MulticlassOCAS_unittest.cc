#include "environments/MutiLabelTestEnvironment.h"
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/MulticlassOCAS.h>

#include <gtest/gtest.h>

#ifdef USE_GPL_SHOGUN

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(MulticlassOCASTest,train)
{
  CMath::init_random(5);
  float64_t C = 1.0;
  std::shared_ptr<GaussianCheckerboard> mockData =
	  MutiLabelTestEnvironment::instance().getMulticlassFixture();

  CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
  CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();
  CMulticlassLabels* ground_truth =
	  (CMulticlassLabels*)mockData->get_labels_test();
  CMulticlassOCAS* mocas = new CMulticlassOCAS(C, train_feats, ground_truth);
  mocas->parallel->set_num_threads(1);
  mocas->set_epsilon(1e-5);
  mocas->train();

  CMulticlassLabels* pred = (CMulticlassLabels*)mocas->apply(test_feats);
  CMulticlassAccuracy evaluate = CMulticlassAccuracy();
  float64_t result = evaluate.evaluate(pred, ground_truth);
  EXPECT_GT(result, 0.99);

  SG_UNREF(mocas);
  SG_UNREF(pred);
}
#endif // HAVE_LAPACK
#endif //USE_GPL_SHOGUN
