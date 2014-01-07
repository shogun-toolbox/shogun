/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <base/init.h>
#include <features/DenseFeatures.h>
#include <labels/MulticlassLabels.h>
#include <evaluation/CrossValidation.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <evaluation/MulticlassAccuracy.h>
#include <modelselection/ModelSelection.h>
#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>
#include <kernel/GaussianKernel.h>
#include <multiclass/MulticlassLibSVM.h>
#include <modelselection/GridSearchModelSelection.h>
#include <mathematics/Math.h>

using namespace shogun;

CModelSelectionParameters* build_param_tree(CKernel* kernel)
{
	CModelSelectionParameters * root=new CModelSelectionParameters();
	CModelSelectionParameters * c=new CModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(-1.0, 1.0, R_EXP);

	CModelSelectionParameters * params_kernel=new CModelSelectionParameters("kernel", kernel);
	root->append_child(params_kernel);
	CModelSelectionParameters * params_kernel_width=new CModelSelectionParameters("width");
	params_kernel_width->build_values(-1.0, 1.0, R_EXP);
	params_kernel->append_child(params_kernel_width);

	return root;
}

void test()
{
	/* number of classes is dimension of data here to have some easy multiclass
	 * structure  */
	const unsigned int num_vectors=50;
	const unsigned int dim_vectors=3;
	// Heiko: increase number of classes and things will fail :(
	// Sergey: the special buggy case of 3 classes was hopefully fixed

	float64_t distance=5;

	/* create data: some easy multiclass data */
	SGMatrix<float64_t> feat=SGMatrix<float64_t>(dim_vectors, num_vectors);
	SGVector<float64_t> lab(num_vectors);
	for (index_t j=0; j<feat.num_cols; ++j)
	{
		lab[j]=j%dim_vectors;

		for (index_t i=0; i<feat.num_rows; ++i)
			feat(i, j)=CMath::randn_double();

		/* make sure classes are (alomst) linearly seperable against each other */
		feat(lab[j],j)+=distance;
	}

	/* shogun representation of above data */
	CDenseFeatures<float64_t> * cfeatures=new CDenseFeatures<float64_t>(feat);
	CMulticlassLabels * clabels=new CMulticlassLabels(lab);

	float64_t sigma=2;
	CGaussianKernel* kernel=new CGaussianKernel(10, sigma);

	const float C=10.;
	CMulticlassLibSVM* cmachine=new CMulticlassLibSVM(C, kernel, clabels);

	CMulticlassAccuracy * eval_crit=new CMulticlassAccuracy();

	/* k-fold stratified x-validation */
	index_t k=3;
	CStratifiedCrossValidationSplitting * splitting=
			new CStratifiedCrossValidationSplitting(clabels, k);

	CCrossValidation * cross=new CCrossValidation(cmachine, cfeatures, clabels,
			splitting, eval_crit);
	cross->set_num_runs(10);
	cross->set_conf_int_alpha(0.05);

	/* create peramters for model selection */
	CModelSelectionParameters* root=build_param_tree(kernel);

	CGridSearchModelSelection * model_selection=new CGridSearchModelSelection(
			cross, root);
	bool print_state=true;
	CParameterCombination * params=model_selection->select_model(print_state);
	SG_SPRINT("best combination\n");
	params->print_tree();

	/* clean up memory */
	SG_UNREF(model_selection);
	SG_UNREF(params);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test();

	exit_shogun();

	return 0;
}
