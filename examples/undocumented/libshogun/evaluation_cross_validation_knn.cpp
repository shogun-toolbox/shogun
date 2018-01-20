/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soumyajit De, Björn Esser
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/CSVFile.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/distance/EuclideanDistance.h>


using namespace shogun;


// Prepare to read a file for the training data
const char fname_feats[]  = "../data/fm_train_real.dat";
const char fname_labels[] = "../data/label_train_multiclass.dat";

void test_cross_validation()
{
	index_t k =4;
	/* dense features from matrix */
	CCSVFile* feature_file = new CCSVFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);
	SG_UNREF(feature_file);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	/* labels from vector */
	CCSVFile* label_file = new CCSVFile(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);
	SG_UNREF(label_file);

	CMulticlassLabels* labels=new CMulticlassLabels(label_vec);
	SG_REF(labels);

	/* create knn */
	CEuclideanDistance* distance = new CEuclideanDistance(features, features);
	CKNN* knn=new CKNN (k, distance, labels);

	/* train and output */
	knn->train(features);
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(knn->apply(features));
	for (index_t i=0; i<features->get_num_vectors(); ++i)
		SG_SPRINT("i=%d, class=%f,\n", i, output->get_label(i));

	/* evaluation criterion */
	CMulticlassAccuracy* eval_crit = new CMulticlassAccuracy ();

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);
	SG_UNREF(output);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=5;
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	CCrossValidation* cross=new CCrossValidation(knn, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(1);
//	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	result->print_result();

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
	SG_UNREF(labels);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test_cross_validation();

	exit_shogun();

	return 0;
}

