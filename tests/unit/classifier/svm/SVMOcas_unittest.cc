#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <gtest/gtest.h>

#include <iostream>

using namespace shogun;

TEST(SVMOcasTest,train)
{
	index_t num_samples = 100, dim = 10;
	float64_t mean_shift = 1.0;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_mean_data(num_samples, dim, mean_shift);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(100), test_idx(100);
	SGVector<float64_t> labels(100);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < 100) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CSVMOcas* ocas = new CSVMOcas(1.0, train_feats, ground_truth);
	ocas->train();
	
	CLabels* pred = ocas->apply(test_feats);
	CROCEvaluation roc;
	float64_t err = CMath::abs(roc.evaluate(pred, ground_truth)-0.7684);
	EXPECT_TRUE(err < 10E-12);

	SG_UNREF(ocas);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
}

