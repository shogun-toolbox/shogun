/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

CModelSelectionParameters* create_param_tree()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* c=new CModelSelectionParameters("C1");
	root->append_child(c);
	c->build_values(1.0, 2.0, R_EXP);

	CGaussianKernel* gaussian_kernel=new CGaussianKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	gaussian_kernel->print_modsel_params();

	CModelSelectionParameters* param_gaussian_kernel=
			new CModelSelectionParameters("kernel", gaussian_kernel);

	root->append_child(param_gaussian_kernel);

	CModelSelectionParameters* param_gaussian_kernel_width=
			new CModelSelectionParameters("log_width");
	param_gaussian_kernel_width->build_values(0.0, 0.5*CMath::log(2.0), R_LINEAR);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	return root;
}

void apply_parameter_tree(CDynamicObjectArray* combinations)
{
	/* create some data */
	SGMatrix<float64_t> matrix(2,3);
	for (index_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	/* create three 2-dimensional vectors
	 * to avoid deleting these, REF now and UNREF when finished */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);
	SG_REF(features);

	/* create three labels, will be handed to svm and automaticall deleted */
	CBinaryLabels* labels=new CBinaryLabels(3);
	SG_REF(labels);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);

	/* create libsvm with C=10 and train */
	CLibSVM* svm=new CLibSVM();
	SG_REF(svm);
	svm->set_labels(labels);

	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		SG_SPRINT("applying:\n");
		CParameterCombination* current_combination=(CParameterCombination*)
				combinations->get_element(i);
		current_combination->print_tree();
		Parameter* current_parameters=svm->m_parameters;
		current_combination->apply_to_modsel_parameter(current_parameters);
		SG_UNREF(current_combination);

		/* get kernel to set features, get_kernel SG_REF's the kernel */
		CKernel* kernel=svm->get_kernel();
		kernel->init(features, features);

		svm->train();

		/* classify on training examples */
		for (index_t j=0; j<3; j++)
			SG_SPRINT("output[%d]=%f\n", j, svm->apply_one(j));

		/* unset features and SG_UNREF kernel */
		kernel->cleanup();
		SG_UNREF(kernel);

		SG_SPRINT("----------------\n\n");
	}

	/* free up memory */
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(svm);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* create example tree */
	CModelSelectionParameters* tree=create_param_tree();
	tree->print_tree();
	SG_SPRINT("----------------------------------\n");

	/* build combinations of parameter trees */
	CDynamicObjectArray* combinations=tree->get_combinations();

	apply_parameter_tree(combinations);

	/* print and directly delete them all */
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* combination=(CParameterCombination*)
				combinations->get_element(i);
		SG_UNREF(combination);
	}

	SG_UNREF(combinations);

	/* delete example tree (after processing of combinations because CSGObject
	 * (namely the kernel) of the tree is SG_UNREF'ed (and not REF'ed anywhere
	 * else) */
	SG_UNREF(tree);

	exit_shogun();

	return 0;
}
