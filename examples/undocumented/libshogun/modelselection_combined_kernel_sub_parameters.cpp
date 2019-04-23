/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Roman Votyakov, Heiko Strathmann, Soumyajit De, 
 *          Evangelos Anagnostopoulos
 */

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
ModelSelectionParameters* build_combined_kernel_parameter_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	/* kernel a should be Gaussian with certain parameters
	 * kernel b should be polynomial with certain parameters
	 * This will create a list of combined kernels with all parameter combinations
	 * All List instances here do reference counting (also the combine_kernels
	 * method of CombinedKernel
	 */
	List* kernels_a=new List(true);
	List* kernels_b=new List(true);

	int32_t cache_size=10;
	kernels_a->append_element(new GaussianKernel(cache_size, 2));
	kernels_a->append_element(new GaussianKernel(cache_size, 4));

	kernels_b->append_element(new CPolyKernel(cache_size, 4));
	kernels_b->append_element(new CPolyKernel(cache_size, 2));

	List* kernel_list=new List();
	kernel_list->append_element(kernels_a);
	kernel_list->append_element(kernels_b);

	List* combinations=CombinedKernel::combine_kernels(kernel_list);

	/* add all created combined kernels to parameters tree */

	/* cast is safe since the above method guarantees the type */
	CombinedKernel* current=(CombinedKernel*)(combinations->get_first_element());
	SG_SPRINT("combined kernel combinations:\n");
	index_t i=0;
	while (current)
	{
		/* print out current kernel's subkernels */
		SG_SPRINT("combined kernel %d:\n", i++);
		GaussianKernel* gaussian=(GaussianKernel*)current->get_kernel(0);
		CPolyKernel* poly=(CPolyKernel*)current->get_kernel(1);
		SG_SPRINT("kernel_a type: %s\n", poly->get_name());
		SG_SPRINT("kernel_b type: %s\n", gaussian->get_name());
		SG_SPRINT("kernel_a parameter: %d\n", poly->get_degree());
		SG_SPRINT("kernel_b parameter: %f\n", gaussian->get_width());

		ModelSelectionParameters* param_kernel=
					new ModelSelectionParameters("kernel", current);
		root->append_child(param_kernel);

		current=(CombinedKernel*)(combinations->get_next_element());
	}


	return root;
}

void modelselection_combined_kernel()
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

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create svm */
	MKL* classifier=new MKLClassification(new CLibSVM());
	classifier->set_interleaved_optimization_enabled(false);

	/* splitting strategy */
	StratifiedCrossValidationSplitting* splitting_strategy=
			new StratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	ContingencyTableEvaluation* evaluation_criterium=
			new ContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CrossValidation* cross=new CrossValidation(classifier, comb_features,
												labels, splitting_strategy,
												evaluation_criterium);
	cross->set_num_runs(1);
	/* TODO: remove this once locking is fixed for combined kernels */
	cross->set_autolock(false);

	ModelSelectionParameters* param_tree=build_combined_kernel_parameter_tree();
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
	CombinedKernel* kernel=(CombinedKernel*)classifier->get_kernel();
	GaussianKernel* gaussian=(GaussianKernel*)kernel->get_kernel(0);
	CPolyKernel* poly=(CPolyKernel*)kernel->get_kernel(1);
	SG_SPRINT("gaussian width: %f\n", gaussian->get_width());
	SG_SPRINT("poly degree: %d\n", poly->get_degree());

	/* larger number of runs to have tighter confidence intervals */
	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.01);
	CrossValidationResult* result=(CrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	/* clean up destroy result parameter */
}

int main(int argc, char **argv)
{
	modelselection_combined_kernel();

	return 0;
}
