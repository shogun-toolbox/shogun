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
	c1->set_range(-5, 5, R_EXP);

//	CModelSelectionParameters* c2=new CModelSelectionParameters("C2");
//	root->append_child(c2);
//	c2->set_range(-5, 5, R_EXP);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	int32_t num_subsets=5;
	int32_t num_features=11;

	/* create some data */
	float64_t* matrix=new float64_t[num_features*2];
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

	/* model parameter selection, deletion is handled by modsel class (SG_UNREF) */
	CModelSelectionParameters* param_tree=create_param_tree();

	/* this is on the stack and handles all of the above structures in memory */
	CGridSearchModelSelection grid_search(param_tree, cross);

	float64_t result;
	ParameterCombination* best_combination=grid_search.select_model(result);
	SG_SPRINT("best parameter(s):\n");
	best_combination->print();
	SG_SPRINT("result: %f\n", result);

	/* clean up destroy result parameter */
	best_combination->destroy(true, true);

	SG_SPRINT("\nEND\n");
	exit_shogun();

	return 0;
}
