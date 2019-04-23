/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/evaluation/StructuredAccuracy.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/structure/MultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/PrimalMosekSOSVM.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void load_data(const char * file_name,
               SGMatrix<float64_t> &feats_matrix,
               int32_t &dim_feat,
               int32_t &num_samples,
               SGVector<int32_t> * &multilabels,
               int32_t &num_classes)
{
	CLibSVMFile * file = new CLibSVMFile(file_name);
	ASSERT(file != NULL);

	SGSparseVector<float64_t> * feats;
	SGVector<float64_t> * labels;

	file->get_sparse_matrix(
	        feats,
	        dim_feat,
	        num_samples,
	        labels,
	        num_classes);

	feats_matrix = SGMatrix<float64_t>(dim_feat, num_samples);

	/** preparation of data for multilabel model */
	for (index_t i = 0; i < num_samples; i++)
	{
		SGSparseVector<float64_t> feat_sample = feats[i];

		for (index_t j = 0; j < dim_feat; j++)
			feats_matrix[i * dim_feat + j] = feat_sample.get_feature(j);
	}

	multilabels = SG_MALLOC(SGVector<int32_t>, num_samples);

	for (index_t i = 0; i < num_samples; i++)
	{
		SGVector<float64_t> label_sample = labels[i];
		SGVector<int32_t> multilabel_sample(label_sample.vlen);

		for (index_t j = 0; j < label_sample.vlen; j++)
			multilabel_sample[j] = label_sample[j];

		Math::qsort(multilabel_sample);

		multilabels[i] = multilabel_sample;
	}

	SG_FREE(feats);
	SG_FREE(labels);
}

int main(int argc, char ** argv)
{
	env()->io()->set_loglevel(MSG_DEBUG);

	const char train_file_name[] = "../../../../data/multilabel/yeast_train.svm";
	const char test_file_name[] = "../../../../data/multilabel/yeast_test.svm";

	SGMatrix<float64_t> feats_matrix;
	SGVector<int32_t> * multilabels;
	int32_t dim_feat;
	int32_t num_samples;
	int32_t num_classes;

	load_data(
	        train_file_name,
	        feats_matrix,
	        dim_feat,
	        num_samples,
	        multilabels,
	        num_classes);

	SG_SPRINT("Number of samples    =  %d\n", num_samples);
	SG_SPRINT("Dimension of feature =  %d\n", dim_feat);
	SG_SPRINT("Number of classes    =  %d\n", num_classes);

	SG_SPRINT("-------------------------------------------\n");

	CMultilabelSOLabels * mlabels = new CMultilabelSOLabels(num_samples,
	                num_classes);
	mlabels->set_sparse_labels(multilabels);

	SparseFeatures<float64_t> * features = new SparseFeatures<float64_t>(
	        feats_matrix);

	MultilabelModel * model = new MultilabelModel(features, mlabels);

	CStochasticSOSVM * sgd = new CStochasticSOSVM(model, mlabels);

	CDualLibQPBMSOSVM * bundle = new CDualLibQPBMSOSVM(model, mlabels, 100);
	bundle->set_verbose(false);

	CPrimalMosekSOSVM * sosvm = new CPrimalMosekSOSVM(model, mlabels);

	Time * start = new Time();
	sgd->train();
	float64_t t1 = start->cur_time_diff(false);
	bundle->train();
	float64_t t2 = start->cur_time_diff(false);
	sosvm->train();
	float64_t t3 = start->cur_time_diff(false);

	SG_SPRINT(">>> Time taken for training using %s = %f\n", sgd->get_name(),
	          t1);
	SG_SPRINT(">>> Time taken for training using %s = %f\n", bundle->get_name(),
	          t2 - t1);
	SG_SPRINT(">>> Time taken for learning using %s = %f\n", sosvm->get_name(),
	          t3 - t2);

	SGMatrix<float64_t> test_feats_matrix;
	SGVector<int32_t> * test_multilabels;

	load_data(
	        test_file_name,
	        test_feats_matrix,
	        dim_feat,
	        num_samples,
	        test_multilabels,
	        num_classes);

	SparseFeatures<float64_t> * test_features = new SparseFeatures<float64_t>(
	        test_feats_matrix);

	CMultilabelSOLabels * test_labels = new CMultilabelSOLabels(num_samples,
	                num_classes);
	test_labels->set_sparse_labels(test_multilabels);

	StructuredLabels* out = sgd->apply(test_features)->as<StructuredLabels>();

	StructuredLabels* bout =
	    bundle->apply(test_features)->as<StructuredLabels>();

	StructuredLabels* sout =
	    sosvm->apply(test_features)->as<StructuredLabels>();

	StructuredAccuracy * evaluator = new StructuredAccuracy();
	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          sgd->get_name(), evaluator->evaluate(out, test_labels));

	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          bundle->get_name(), evaluator->evaluate(bout, test_labels));

	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          sosvm->get_name(), evaluator->evaluate(sout, test_labels));

	SG_FREE(multilabels);
	SG_FREE(test_multilabels);

	return 0;
}

