/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Jacob Walker, Viktor Gal, 
 *          Evgeniy Andreev, Soumyajit De, Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/CSVFile.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;


// Prepare to read a file for the training data
const char fname_feats[]  = "../data/fm_train_real.dat";
const char fname_labels[] = "../data/label_train_multiclass.dat";

void test_cross_validation()
{
	/* dense features from matrix */
	auto feature_file = std::make_shared<CSVFile>(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);

	auto features=std::make_shared<DenseFeatures<float64_t>>(mat);

	/* labels from vector */
	auto label_file = std::make_shared<CSVFile>(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);

	auto labels=std::make_shared<MulticlassLabels>(label_vec);

	/* create svm via libsvm */
	float64_t svm_C=10;
	float64_t svm_eps=0.0001;
	auto svm=std::make_shared<MulticlassLibLinear>(svm_C, features, labels);
	svm->set_epsilon(svm_eps);

	/* train and output */
	svm->train(features);
	auto output = svm->apply(features)->as<MulticlassLabels>();
	for (index_t i=0; i<features->get_num_vectors(); ++i)
		SG_SPRINT("i=%d, class=%f,\n", i, output->get_label(i));

	/* evaluation criterion */
	auto eval_crit = std::make_shared<MulticlassAccuracy>();

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=5;
	auto splitting=
			std::make_shared<StratifiedCrossValidationSplitting>(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	auto cross=std::make_shared<CrossValidation>(svm, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(1);
//	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	auto result=cross->evaluate()->as<CrossValidationResult>();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	result->print_result();

	/* clean up */
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test_cross_validation();

	exit_shogun();

	return 0;
}

