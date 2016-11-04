/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
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

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

CModelSelectionParameters* create_param_tree()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* c1=new CModelSelectionParameters("C1");
	root->append_child(c1);
	c1->build_values(1.0, 2.0, R_EXP);

	CModelSelectionParameters* c2=new CModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(1.0, 2.0, R_EXP);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	ds_kernel->print_modsel_params();


	CModelSelectionParameters* param_ds_kernel=
			new CModelSelectionParameters("kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	CModelSelectionParameters* ds_kernel_delta=
			new CModelSelectionParameters("delta");
	ds_kernel_delta->build_values(1, 2, R_LINEAR);
	param_ds_kernel->append_child(ds_kernel_delta);

	CModelSelectionParameters* ds_kernel_theta=
			new CModelSelectionParameters("theta");
	ds_kernel_theta->build_values(1, 2, R_LINEAR);
	param_ds_kernel->append_child(ds_kernel_theta);

	return root;
}


int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	index_t num_strings=10;
	index_t max_string_length=20;
	index_t min_string_length=max_string_length/2;
	index_t num_subsets=num_strings/3;

	SGStringList<char> strings(num_strings, max_string_length);

	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=CMath::random(min_string_length, max_string_length);
		SGString<char> current(len);

		SG_SPRINT("string %i: \"", i);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j=0; j<len; ++j)
		{
			current.string[j]=(char)CMath::random('A', 'Z');

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
	CBinaryLabels* labels=new CBinaryLabels(num_strings);
	for (index_t i=0; i<num_strings; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create svm classifier */
	CLibSVM* classifier=new CLibSVM();

	/* splitting strategy */
	CStratifiedCrossValidationSplitting* splitting_strategy=
			new CStratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	CContingencyTableEvaluation* evaluation_criterium=
			new CContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CCrossValidation* cross=new CCrossValidation(classifier, features, labels,
			splitting_strategy, evaluation_criterium);
	cross->set_num_runs(2);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	classifier->print_modsel_params();

	/* model parameter selection, deletion is handled by modsel class (SG_UNREF) */
	CModelSelectionParameters* param_tree=create_param_tree();
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
	classifier->data_lock(labels, features);
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("result: ");
	result->print_result();

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);

	exit_shogun();

	return 0;
}
