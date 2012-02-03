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
#include <shogun/lib/config.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/features/Labels.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/svm/LibLinear.h>

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
	c1->build_values(-2.0, 2.0, R_EXP);

	CModelSelectionParameters* c2=new CModelSelectionParameters("C2");
	root->append_child(c2);
	c2->build_values(-2.0, 2.0, R_EXP);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

#ifdef HAVE_LAPACK
	int32_t num_subsets=5;
	int32_t num_features=11;

	/* create some data */
	float64_t* matrix=SG_MALLOC(float64_t, num_features*2);
	for (int32_t i=0; i<num_features*2; i++)
		matrix[i]=i;

	/* create num_feautres 2-dimensional vectors */
	CSimpleFeatures<float64_t>* features=new CSimpleFeatures<float64_t> ();
	features->set_feature_matrix(matrix, 2, num_features);

	/* create three labels */
	CLabels* labels=new CLabels(num_features);
	for (index_t i=0; i<num_features; ++i)
		labels->set_label(i, i%2==0 ? 1 : -1);

	/* create linear classifier (use -s 2 option to avoid warnings) */
	CLibLinear* classifier=new CLibLinear(L2R_L2LOSS_SVC);

	/* splitting strategy */
	CStratifiedCrossValidationSplitting* splitting_strategy=
			new CStratifiedCrossValidationSplitting(labels, num_subsets);

	/* accuracy evaluation */
	CContingencyTableEvaluation* evaluation_criterium=
			new CContingencyTableEvaluation(ACCURACY);

	/* cross validation class for evaluation in model selection */
	CCrossValidation* cross=new CCrossValidation(classifier, features, labels,
			splitting_strategy, evaluation_criterium);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	classifier->print_modsel_params();


	/* model parameter selection, deletion is handled by modsel class (SG_UNREF) */
	CModelSelectionParameters* param_tree=create_param_tree();
	param_tree->print_tree();

	/* handles all of the above structures in memory */
	CGridSearchModelSelection* grid_search=new CGridSearchModelSelection(
			param_tree, cross);

	CParameterCombination* best_combination=grid_search->select_model();
	SG_SPRINT("best parameter(s):\n");
	best_combination->print_tree();

	best_combination->apply_to_machine(classifier);
	CrossValidationResult result=cross->evaluate();
	result.print_result();

	/* clean up */
	SG_UNREF(best_combination);
	SG_UNREF(grid_search);
#endif // HAVE_LAPACK
	exit_shogun();

	return 0;
}

