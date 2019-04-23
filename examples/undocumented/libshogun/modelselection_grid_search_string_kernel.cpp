/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Jacob Walker, Soeren Sonnenburg, Soumyajit De, 
 *          Sergey Lisitsyn, Roman Votyakov
 */

#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/lib/SGStringList.h>


using namespace shogun;

ModelSelectionParameters* create_param_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c1=new ModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(1.0, 2.0, R_EXP);

	ModelSelectionParameters* c2=new ModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(1.0, 2.0, R_EXP);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	ds_kernel->print_modsel_params();


	ModelSelectionParameters* param_ds_kernel=
			new ModelSelectionParameters("kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	ModelSelectionParameters* ds_kernel_delta=
			new ModelSelectionParameters("delta");
	ds_kernel_delta->build_values(1, 2, R_LINEAR);
	param_ds_kernel->append_child(ds_kernel_delta);

	ModelSelectionParameters* ds_kernel_theta=
			new ModelSelectionParameters("theta");
	ds_kernel_theta->build_values(1, 2, R_LINEAR);
	param_ds_kernel->append_child(ds_kernel_theta);

	return root;
}


int main(int argc, char **argv)
{
	index_t num_strings=10;
	index_t max_string_length=20;
	index_t min_string_length=max_string_length/2;
	index_t num_subsets=num_strings/3;

	SGStringList<char> strings(num_strings, max_string_length);

	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=Math::random(min_string_length, max_string_length);
		SGString<char> current(len);

		SG_SPRINT("string %i: \"", i);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j=0; j<len; ++j)
		{
			current.string[j]=(char)Math::random('A', 'Z');

			char* string=new char[2];
			string[0]=current.string[j];
			string[1]='\0';
			SG_SPRINT("%s", string);
			delete[] string;
		}
		SG_SPRINT("\"\n");

		strings.strings[i]=current;
	}

	/* create num_feautres 2-dimensional vectors */
	CStringFeatures<char>* features=new CStringFeatures<char>(strings, ALPHANUM);

	/* create labels, two classes */
	BinaryLabels* labels=new BinaryLabels(num_strings);
	for (index_t i=0; i<num_strings; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create svm classifier */
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
	cross->set_num_runs(2);

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
	CrossValidationResult* result=(CrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	/* clean up */

	return 0;
}
