/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Michele Mazzoni, Roman Votyakov, Heiko Strathmann, Soumyajit De,
 *          Viktor Gal
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/distance/MinkowskiMetric.h>

using namespace shogun;

ModelSelectionParameters* create_param_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c1=new ModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(-1.0, 1.0, R_EXP);

	ModelSelectionParameters* c2=new ModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-1.0, 1.0, R_EXP);

	CombinedKernel* kernel1=new CombinedKernel();
	kernel1->append_kernel(new GaussianKernel(10, 2));
	kernel1->append_kernel(new GaussianKernel(10, 3));
	kernel1->append_kernel(new GaussianKernel(10, 4));

	ModelSelectionParameters* param_kernel1=new ModelSelectionParameters(
			"kernel", kernel1);
	root->append_child(param_kernel1);

	CombinedKernel* kernel2=new CombinedKernel();
	kernel2->append_kernel(new GaussianKernel(10, 20));
	kernel2->append_kernel(new GaussianKernel(10, 30));
	kernel2->append_kernel(new GaussianKernel(10, 40));

	ModelSelectionParameters* param_kernel2=new ModelSelectionParameters(
			"kernel", kernel2);
	root->append_child(param_kernel2);

	return root;
}

/** Demonstrates the MKL modelselection bug with SVMLight. See comments how to reproduce */
void test()
{
	int32_t num_subsets=3;
	int32_t num_vectors=20;
	int32_t dim_vectors=3;

	/* create some data and labels */
	SGMatrix<float64_t> matrix(dim_vectors, num_vectors);
	BinaryLabels* labels=new BinaryLabels(num_vectors);
	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
		matrix.matrix[i]=Math::randn_double();

	/* create num_feautres 2-dimensional vectors */
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);

	/* create combined features */
	CombinedFeatures* comb_features=new CombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* works */
	MKLClassification* classifier=new MKLClassification(new CLibSVM());
	classifier->set_interleaved_optimization_enabled(false);

	/* the above plus this does not work (interleaved only with SVMLight)*/
//	classifier->set_interleaved_optimization_enabled(true);

	/* However, SVMLight does not work */
//	MKLClassification* classifier=new MKLClassification(new SVMLight());
//	 /* any of those */
//	classifier->set_interleaved_optimization_enabled(false);
//	classifier->set_interleaved_optimization_enabled(true);

	/* splitting strategy */
	StratifiedCrossValidationSplitting* splitting_strategy=
			new StratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	ContingencyTableEvaluation* evaluation_criterium=
			new ContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CrossValidation* cross=new CrossValidation(classifier, comb_features,
			labels, splitting_strategy, evaluation_criterium);
	cross->set_num_runs(1);
	/* TODO: remove this once locking is fixed for combined kernels */
	cross->set_autolock(false);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	classifier->print_modsel_params();

	ModelSelectionParameters* param_tree=create_param_tree();
	param_tree->print_tree();

	/* handles all of the above structures in memory */
	CGridSearchModelSelection* grid_search=new CGridSearchModelSelection(
			cross, param_tree);

	bool print_state=true;
	CParameterCombination* best_combination=grid_search->select_model(
			print_state);
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	/* larger number of runs to have tighter confidence intervals */
	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.01);
	EvaluationResult* result=cross->evaluate();
	SG_SPRINT("result: ");
	result->print_result();

	/* clean up destroy result parameter */
}

int main(int argc, char **argv)
{
	env()->io()->set_loglevel(MSG_INFO);
	test();

	return 0;
}
