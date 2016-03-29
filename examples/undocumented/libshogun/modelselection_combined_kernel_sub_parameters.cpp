/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/CombinedKernel.h>

using namespace shogun;

/** Creates a bunch of combined kernels with different sub-parameters.
 * This can be used for modelselection of subkernel parameters of combined
 * kernels
 */
CModelSelectionParameters* build_combined_kernel_parameter_tree()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	/* kernel a should be Gaussian with certain parameters
	 * kernel b should be polynomial with certain parameters
	 * This will create a list of combined kernels with all parameter combinations
	 * All CList instances here do reference counting (also the combine_kernels
	 * method of CCombinedKernel
	 */
	CList* kernels_a=new CList(true);
	CList* kernels_b=new CList(true);

	int32_t cache_size=10;
	kernels_a->append_element(new CGaussianKernel(cache_size, 2));
	kernels_a->append_element(new CGaussianKernel(cache_size, 4));

	kernels_b->append_element(new CPolyKernel(cache_size, 4));
	kernels_b->append_element(new CPolyKernel(cache_size, 2));

	CList* kernel_list=new CList();
	kernel_list->append_element(kernels_a);
	kernel_list->append_element(kernels_b);

	CList* combinations=CCombinedKernel::combine_kernels(kernel_list);

	/* add all created combined kernels to parameters tree */

	/* cast is safe since the above method guarantees the type */
	CCombinedKernel* current=(CCombinedKernel*)(combinations->get_first_element());
	SG_SPRINT("combined kernel combinations:\n");
	index_t i=0;
	while (current)
	{
		/* print out current kernel's subkernels */
		SG_SPRINT("combined kernel %d:\n", i++);
		CGaussianKernel* gaussian=(CGaussianKernel*)current->get_kernel(0);
		CPolyKernel* poly=(CPolyKernel*)current->get_kernel(1);
		SG_SPRINT("kernel_a type: %s\n", poly->get_name());
		SG_SPRINT("kernel_b type: %s\n", gaussian->get_name());
		SG_SPRINT("kernel_a parameter: %d\n", poly->get_degree());
		SG_SPRINT("kernel_b parameter: %f\n", gaussian->get_width());
		SG_UNREF(poly);
		SG_UNREF(gaussian);

		CModelSelectionParameters* param_kernel=
					new CModelSelectionParameters("kernel", current);
		root->append_child(param_kernel);

		SG_UNREF(current);
		current=(CCombinedKernel*)(combinations->get_next_element());
	}

	SG_UNREF(combinations);
	SG_UNREF(kernel_list);
	SG_UNREF(kernels_a);
	SG_UNREF(kernels_b);

	return root;
}

void modelselection_combined_kernel()
{
	int32_t num_subsets=3;
	int32_t num_vectors=20;
	int32_t dim_vectors=3;

	/* create some data and labels */
	SGMatrix<float64_t> matrix(dim_vectors, num_vectors);
	CBinaryLabels* labels=new CBinaryLabels(num_vectors);

	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
		matrix.matrix[i]=CMath::randn_double();

	/* create num_feautres 2-dimensional vectors */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);

	/* create combined features */
	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create svm */
	CMKL* classifier=new CMKLClassification(new CLibSVM());
	classifier->set_interleaved_optimization_enabled(false);

	/* splitting strategy */
	CStratifiedCrossValidationSplitting* splitting_strategy=
			new CStratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	CContingencyTableEvaluation* evaluation_criterium=
			new CContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CCrossValidation* cross=new CCrossValidation(classifier, comb_features,
												labels, splitting_strategy,
												evaluation_criterium);
	cross->set_num_runs(1);
	/* TODO: remove this once locking is fixed for combined kernels */
	cross->set_autolock(false);

	/* model parameter selection, deletion is handled by modsel class (SG_UNREF) */
	CModelSelectionParameters* param_tree=build_combined_kernel_parameter_tree();
	param_tree->print_tree();

	/* handles all of the above structures in memory */
	CGridSearchModelSelection* grid_search=new CGridSearchModelSelection(
			cross, param_tree);

	bool print_state=true;
	CParameterCombination* best_combination=grid_search->select_model(
			print_state);
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	/* print subkernel parameters, I know what the subkernel types are here */
	CCombinedKernel* kernel=(CCombinedKernel*)classifier->get_kernel();
	CGaussianKernel* gaussian=(CGaussianKernel*)kernel->get_kernel(0);
	CPolyKernel* poly=(CPolyKernel*)kernel->get_kernel(1);
	SG_SPRINT("gaussian width: %f\n", gaussian->get_width());
	SG_SPRINT("poly degree: %d\n", poly->get_degree());
	SG_UNREF(kernel);
	SG_UNREF(gaussian);
	SG_UNREF(poly);

	/* larger number of runs to have tighter confidence intervals */
	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.01);
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	/* clean up destroy result parameter */
	SG_UNREF(result);
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	modelselection_combined_kernel();

	exit_shogun();

	return 0;
}
