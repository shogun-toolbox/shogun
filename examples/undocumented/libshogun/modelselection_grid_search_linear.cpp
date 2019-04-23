/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Jacob Walker, Roman Votyakov, 
 *          Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/lib/config.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibLinear.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

ModelSelectionParameters* create_param_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c1=new ModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(-2.0, 2.0, R_EXP);

	ModelSelectionParameters* c2=new ModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-2.0, 2.0, R_EXP);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

#ifdef HAVE_LAPACK
	int32_t num_subsets=5;
	int32_t num_vectors=11;

	/* create some data */
	SGMatrix<float64_t> matrix(2, num_vectors);
	for (int32_t i=0; i<num_vectors*2; i++)
		matrix.matrix[i]=i;

	/* create num_feautres 2-dimensional vectors */
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);

	/* create three labels */
	BinaryLabels* labels=new BinaryLabels(num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create linear classifier (use -s 2 option to avoid warnings) */
	LibLinear* classifier=new LibLinear(L2R_L2LOSS_SVC);

	/* splitting strategy */
	StratifiedCrossValidationSplitting* splitting_strategy=
			new StratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	ContingencyTableEvaluation* evaluation_criterium=
			new ContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CrossValidation* cross=new CrossValidation(classifier, features, labels,
			splitting_strategy, evaluation_criterium);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	classifier->print_modsel_params();

	ModelSelectionParameters* param_tree=create_param_tree();
	param_tree->print_tree();

	/* handles all of the above structures in memory */
	CGridSearchModelSelection* grid_search=new CGridSearchModelSelection(
			cross, param_tree);

	/* set autolocking to false to get rid of warnings */
	cross->set_autolock(false);

	CParameterCombination* best_combination=grid_search->select_model();
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);
	CrossValidationResult* result=(CrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	result->print_result();

	/* clean up */
#endif // HAVE_LAPACK
	exit_shogun();

	return 0;
}
