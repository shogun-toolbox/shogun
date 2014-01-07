#include <multiclass/MulticlassOCAS.h>
#include <features/DataGenerator.h>
#include <features/DenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(MulticlassOCASTest,train)
{
  float64_t C = 1.0;
  index_t num_samples = 50, num_gauss = 3, dim = 3;
  CMath::init_random(5);
  SGMatrix<float64_t> data =
    CDataGenerator::generate_gaussians(num_samples, num_gauss, dim);
  CDenseFeatures<float64_t> features(data);

  index_t set_size = data.num_cols/2;
  SGVector<index_t> train_idx(set_size), test_idx(set_size);
  SGVector<float64_t> labels(set_size);
  for (index_t i = 0, j = 0; i < data.num_cols; ++i)
  {
    if (i % 2 == 0)
      train_idx[j] = i;
    else
      test_idx[j++] = i;

    if (i < data.num_cols/num_gauss)
      labels[i/2] = 0.0;
    else if (i < 2*data.num_cols/num_gauss)
      labels[i/2] = 1.0;
    else
      labels[i/2] = 2.0;
  }

  CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
  CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

  CMulticlassLabels* ground_truth = new CMulticlassLabels(labels);
  CMulticlassOCAS* mocas = new CMulticlassOCAS(C, train_feats, ground_truth);
  mocas->parallel->set_num_threads(1);
  mocas->set_epsilon(1e-5);
  mocas->train();

  CLabels* pred = mocas->apply(test_feats);
  for (int i = 0; i < set_size; ++i)
    EXPECT_EQ(ground_truth->get_label(i), ((CMulticlassLabels*)pred)->get_label(i));

  SG_UNREF(mocas);
  SG_UNREF(train_feats);
  SG_UNREF(test_feats);
  SG_UNREF(pred);
}
#endif // HAVE_LAPACK
