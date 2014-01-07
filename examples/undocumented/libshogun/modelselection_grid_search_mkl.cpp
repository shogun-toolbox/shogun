/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <base/init.h>
#include <evaluation/CrossValidation.h>
#include <evaluation/ContingencyTableEvaluation.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <modelselection/GridSearchModelSelection.h>
#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>
#include <features/Labels.h>
#include <features/DenseFeatures.h>
#include <classifier/mkl/MKLClassification.h>
#include <classifier/svm/LibSVM.h>
#include <kernel/GaussianKernel.h>
#include <kernel/CombinedKernel.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

CModelSelectionParameters* create_param_tree()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* c1=new CModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(-1.0, 1.0, R_EXP);

	CModelSelectionParameters* c2=new CModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-1.0, 1.0, R_EXP);

	CCombinedKernel* kernel1=new CCombinedKernel();
	kernel1->append_kernel(new CGaussianKernel(10, 2));
	kernel1->append_kernel(new CGaussianKernel(10, 3));
	kernel1->append_kernel(new CGaussianKernel(10, 4));

	CModelSelectionParameters* param_kernel1=
			new CModelSelectionParameters("kernel", kernel1);
	root->append_child(param_kernel1);

	CCombinedKernel* kernel2=new CCombinedKernel();
	kernel2->append_kernel(new CGaussianKernel(10, 20));
	kernel2->append_kernel(new CGaussianKernel(10, 30));
	kernel2->append_kernel(new CGaussianKernel(10, 40));

	CModelSelectionParameters* param_kernel2=
			new CModelSelectionParameters("kernel", kernel2);
	root->append_child(param_kernel2);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	sg_io->set_loglevel(MSG_INFO);

	int32_t num_subsets=3;
	int32_t num_vectors=20;
	int32_t dim_vectors=3;

	/* create some data and labels */
	float64_t* matrix=SG_MALLOC(float64_t, num_vectors*dim_vectors);
	CLabels* labels=new CLabels(num_vectors);
	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
		matrix[i]=CMath::randn_double();

	/* create num_feautres 2-dimensional vectors */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
	features->set_feature_matrix(matrix, dim_vectors, num_vectors);

	/* create combined features */
	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* works */
//	/* create svm */
//	CMKLClassification* classifier=new CMKLClassification(new CLibSVM());
//	classifier->set_interleaved_optimization_enabled(false);

	/* create svm */
	CMKLClassification* classifier=new CMKLClassification();

	// both fail:
	//classifier->set_interleaved_optimization_enabled(false);
	classifier->set_interleaved_optimization_enabled(true);

	/* splitting strategy */
	CStratifiedCrossValidationSplitting* splitting_strategy=
			new CStratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	CContingencyTableEvaluation* evaluation_criterium=
			new CContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CCrossValidation* cross=new CCrossValidation(classifier, comb_features,
			labels, splitting_strategy, evaluation_criterium);
	cross->set_num_runs(1);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	classifier->print_modsel_params();

	/* model parameter selection, deletion is handled by modsel class (SG_UNREF) */
	CModelSelectionParameters* param_tree=create_param_tree();
	param_tree->print_tree();

	/* handles all of the above structures in memory */
	CGridSearchModelSelection* grid_search=new CGridSearchModelSelection(
			param_tree, cross);

	bool print_state=true;
	CParameterCombination* best_combination=grid_search->select_model(
			print_state);
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	/* larger number of runs to have tighter confidence intervals */
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.01);
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_ERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result: ");
	result.print_result();

	/* clean up destroy result parameter */
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);

	exit_shogun();

	return 0;
}

