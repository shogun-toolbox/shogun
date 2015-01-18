/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/base/init.h>
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
	SG_REF(file);

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
		{
			feats_matrix[i * dim_feat + j] = feat_sample.get_feature(j);
		}
	}

	multilabels = SG_MALLOC(SGVector<int32_t>, num_samples);

	for (index_t i = 0; i < num_samples; i++)
	{
		SGVector<float64_t> label_sample = labels[i];
		SGVector<int32_t> multilabel_sample(label_sample.vlen);

		for (index_t j = 0; j < label_sample.vlen; j++)
		{
			multilabel_sample[j] = label_sample[j];
		}

		multilabel_sample.qsort();

		multilabels[i] = multilabel_sample;
	}

	SG_UNREF(file);
	SG_FREE(feats);
	SG_FREE(labels);
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

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
	SG_REF(mlabels);
	mlabels->set_sparse_labels(multilabels);

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(
	        feats_matrix);
	SG_REF(features);

	CMultilabelModel * model = new CMultilabelModel(features, mlabels);
	SG_REF(model);

	CStochasticSOSVM * sgd = new CStochasticSOSVM(model, mlabels);
	SG_REF(sgd);

	CDualLibQPBMSOSVM * bundle = new CDualLibQPBMSOSVM(model, mlabels, 100);
	bundle->set_verbose(false);
	SG_REF(bundle);

	CPrimalMosekSOSVM * sosvm = new CPrimalMosekSOSVM(model, mlabels);
	SG_REF(sosvm);

	CTime * start = new CTime();
	SG_REF(start);
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

	CSparseFeatures<float64_t> * test_features = new CSparseFeatures<float64_t>(
	        test_feats_matrix);
	SG_REF(test_features);

	CMultilabelSOLabels * test_labels = new CMultilabelSOLabels(num_samples,
	                num_classes);
	SG_REF(test_labels);
	test_labels->set_sparse_labels(test_multilabels);

	CStructuredLabels * out = CLabelsFactory::to_structured(
	                                  sgd->apply(test_features));

	CStructuredLabels * bout = CLabelsFactory::to_structured(
	                                   bundle->apply(test_features));

	CStructuredLabels * sout = CLabelsFactory::to_structured(
	                                   sosvm->apply(test_features));


	CStructuredAccuracy * evaluator = new CStructuredAccuracy();
	SG_REF(evaluator);
	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          sgd->get_name(), evaluator->evaluate(out, test_labels));

	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          bundle->get_name(), evaluator->evaluate(bout, test_labels));

	SG_SPRINT(">>> Accuracy of multilabel classification using %s = %f\n",
	          sosvm->get_name(), evaluator->evaluate(sout, test_labels));

	SG_UNREF(bout);
	SG_UNREF(bundle);
	SG_UNREF(evaluator);
	SG_UNREF(features);
	SG_UNREF(mlabels);
	SG_UNREF(model);
	SG_UNREF(out);
	SG_UNREF(sgd);
	SG_UNREF(sosvm);
	SG_UNREF(sout);
	SG_UNREF(start);
	SG_UNREF(test_features);
	SG_UNREF(test_labels);
	SG_FREE(multilabels);
	SG_FREE(test_multilabels);

	exit_shogun();
	return 0;
}

