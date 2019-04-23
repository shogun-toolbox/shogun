/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Jacob Walker, Soumyajit De, 
 *          Sergey Lisitsyn, Roman Votyakov, Wu Lin
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
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/mathematics/Math.h>


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
	c1->build_values(-1.0, 1.0, R_EXP);

	ModelSelectionParameters* c2=new ModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-1.0, 1.0, R_EXP);

	GaussianKernel* gaussian_kernel=new GaussianKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	gaussian_kernel->print_modsel_params();

	ModelSelectionParameters* param_gaussian_kernel=
			new ModelSelectionParameters("kernel", gaussian_kernel);
	ModelSelectionParameters* gaussian_kernel_width=
			new ModelSelectionParameters("log_width");
	gaussian_kernel_width->build_values(-std::log(2.0), 0.0, R_LINEAR, 1.0);
	param_gaussian_kernel->append_child(gaussian_kernel_width);
	root->append_child(param_gaussian_kernel);

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	ModelSelectionParameters* param_power_kernel=
	new ModelSelectionParameters("kernel", power_kernel);

	root->append_child(param_power_kernel);

	ModelSelectionParameters* param_power_kernel_degree=
			new ModelSelectionParameters("degree");
	param_power_kernel_degree->build_values(1.0, 2.0, R_LINEAR);
	param_power_kernel->append_child(param_power_kernel_degree);

	CMinkowskiMetric* m_metric=new CMinkowskiMetric(10);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	m_metric->print_modsel_params();

	ModelSelectionParameters* param_power_kernel_metric1=
			new ModelSelectionParameters("distance", m_metric);

	param_power_kernel->append_child(param_power_kernel_metric1);

	ModelSelectionParameters* param_power_kernel_metric1_k=
			new ModelSelectionParameters("k");
	param_power_kernel_metric1_k->build_values(1.0, 2.0, R_LINEAR);
	param_power_kernel_metric1->append_child(param_power_kernel_metric1_k);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

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

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create svm */
	CLibSVM* classifier=new CLibSVM();

	/* splitting strategy */
	StratifiedCrossValidationSplitting* splitting_strategy=
			new StratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	ContingencyTableEvaluation* evaluation_criterium=
			new ContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CrossValidation* cross=new CrossValidation(classifier, features, labels,
			splitting_strategy, evaluation_criterium);
	cross->set_num_runs(1);
	/* note that this automatically is not necessary since done automatically */
	cross->set_autolock(true);

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
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);

	/* larger number of runs to have tighter confidence intervals */
	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.01);
	CrossValidationResult* result=(CrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	/* now again but unlocked */
	cross->set_autolock(true);
	best_combination=grid_search->select_model(print_state);
	best_combination->apply_to_machine(classifier);
	result=(CrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	SG_SPRINT("result (unlocked): ");

	/* clean up destroy result parameter */

	exit_shogun();

	return 0;
}
