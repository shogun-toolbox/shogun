#include <shogun/features/DataGenerator.h>
#include <shogun/transfer/multitask/MultitaskL12LogisticRegression.h>
#include <shogun/transfer/multitask/MultitaskClusteredLogisticRegression.h>
#include <shogun/transfer/multitask/MultitaskTraceLogisticRegression.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>

#include <gtest/gtest.h>

#include <utility>

using namespace shogun;

typedef std::pair<CDotFeatures*, CDotFeatures*> SplittedFeatures;
typedef std::pair<SplittedFeatures, CBinaryLabels*> SplittedDataset;

#ifdef HAVE_LAPACK
SplittedDataset generate_data()
{
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features->copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features->copy_subset(test_idx);

	CBinaryLabels* ground_truth = new CBinaryLabels(labels);
	SG_UNREF(features);

	return SplittedDataset(SplittedFeatures(train_feats, test_feats), ground_truth);
}

TEST(MalsarL12Test, train)
{
	SplittedDataset data = generate_data();

	CTaskGroup* task_group = new CTaskGroup();
	CTask* task = new CTask(0, data.second->get_num_labels());
	task_group->append_task(task);

	CMultitaskL12LogisticRegression* mtlr = new CMultitaskL12LogisticRegression(0.1,0.1,data.first.first,data.second,task_group);
	mtlr->train();
	mtlr->set_features(data.first.second);
	CLabels* output = mtlr->apply();
	SG_UNREF(output);
	SG_UNREF(mtlr);
	SG_UNREF(data.first.first);
	SG_UNREF(data.first.second);
}

TEST(MalsarClusteredTest, train)
{
	SplittedDataset data = generate_data();

	CTaskGroup* task_group = new CTaskGroup();
	CTask* task = new CTask(0, data.second->get_num_labels());
	task_group->append_task(task);

	CMultitaskClusteredLogisticRegression* mtlr = new CMultitaskClusteredLogisticRegression(0.1,0.1,data.first.first,data.second,task_group,1);
	mtlr->train();
	mtlr->set_features(data.first.second);
	CLabels* output = mtlr->apply();
	SG_UNREF(output);
	SG_UNREF(mtlr);
	SG_UNREF(data.first.first);
	SG_UNREF(data.first.second);
}

TEST(MalsarTraceTest, train)
{
	SplittedDataset data = generate_data();

	CTaskGroup* task_group = new CTaskGroup();
	CTask* task = new CTask(0, data.second->get_num_labels());
	task_group->append_task(task);

	CMultitaskTraceLogisticRegression* mtlr = new CMultitaskTraceLogisticRegression(0.1,data.first.first,data.second,task_group);
	mtlr->train();
	mtlr->set_features(data.first.second);
	CLabels* output = mtlr->apply();
	SG_UNREF(output);
	SG_UNREF(mtlr);
	SG_UNREF(data.first.first);
	SG_UNREF(data.first.second);
}
#endif // HAVE_LAPACK

