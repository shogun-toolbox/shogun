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
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/kernel/DistantSegmentsKernel.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

CModelSelectionParameters* create_param_tree()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* kernel=new CModelSelectionParameters("kernel");
	root->append_child(kernel);

	CModelSelectionParameters* c=new CModelSelectionParameters("C");
	root->append_child(c);
	c->set_range(1, 11, R_EXP);

	CPowerKernel* power_kernel=new CPowerKernel();
	CModelSelectionParameters* param_power_kernel=
			new CModelSelectionParameters("kernel", power_kernel);

	kernel->append_child(param_power_kernel);

	CModelSelectionParameters* param_power_kernel_degree=
			new CModelSelectionParameters("degree");
	param_power_kernel_degree->set_range(1, 1, R_EXP);
	param_power_kernel->append_child(param_power_kernel_degree);

	CMinkowskiMetric* m_metric=new CMinkowskiMetric(10);
	CModelSelectionParameters* param_power_kernel_metric1=
			new CModelSelectionParameters("distance", m_metric);

	param_power_kernel->append_child(param_power_kernel_metric1);

	CModelSelectionParameters* param_power_kernel_metric1_k=
			new CModelSelectionParameters("k");
	param_power_kernel_metric1_k->set_range(1, 12, R_LINEAR);
	param_power_kernel_metric1->append_child(param_power_kernel_metric1_k);

	CGaussianKernel* gaussian_kernel=new CGaussianKernel();
	CModelSelectionParameters* param_gaussian_kernel=
			new CModelSelectionParameters("kernel", gaussian_kernel);

	kernel->append_child(param_gaussian_kernel);

	CModelSelectionParameters* param_gaussian_kernel_width=
			new CModelSelectionParameters("width");
	param_gaussian_kernel_width->set_range(1, 2, R_EXP);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();
	CModelSelectionParameters* param_ds_kernel=new CModelSelectionParameters("kernel",
			ds_kernel);

	kernel->append_child(param_ds_kernel);

	CModelSelectionParameters* param_ds_kernel_delta=
			new CModelSelectionParameters("delta");
	param_ds_kernel_delta->set_range(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_delta);

	CModelSelectionParameters* param_ds_kernel_theta=
			new CModelSelectionParameters("theta");
	param_ds_kernel_theta->set_range(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_theta);

	return root;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* create example tree */
	CModelSelectionParameters* tree=create_param_tree();
	tree->print();

	/* build combinations of parameter trees */
	DynArray<ParameterCombination*> combinations;
	tree->get_combinations(combinations);

	/* print and directly delete them all */
	SG_SPRINT("----------------------------------\n");
	for (index_t i=0; i<combinations.get_num_elements(); ++i)
	{
		combinations[i]->print();
		combinations[i]->destroy(true, true);
	}

	/* delete example tree */
	tree->destroy();

	SG_SPRINT("END\n");

	exit_shogun();

	return 0;
}

