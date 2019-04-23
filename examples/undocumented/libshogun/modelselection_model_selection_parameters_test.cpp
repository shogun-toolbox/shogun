/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Wu Lin
 */

#include <shogun/base/init.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

ModelSelectionParameters* build_complex_example_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c=new ModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(1.0, 1.0, R_EXP);

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	ModelSelectionParameters* param_power_kernel=
			new ModelSelectionParameters("kernel", power_kernel);

	root->append_child(param_power_kernel);

	ModelSelectionParameters* param_power_kernel_degree=
			new ModelSelectionParameters("degree");
	param_power_kernel_degree->build_values(1.0, 1.0, R_EXP);
	param_power_kernel->append_child(param_power_kernel_degree);

	CMinkowskiMetric* m_metric=new CMinkowskiMetric(10);
	ModelSelectionParameters* param_power_kernel_metric1=
			new ModelSelectionParameters("distance", m_metric);

	param_power_kernel->append_child(param_power_kernel_metric1);

	ModelSelectionParameters* param_power_kernel_metric1_k=
			new ModelSelectionParameters("k");
	param_power_kernel_metric1_k->build_values(1.0, 12.0, R_LINEAR);
	param_power_kernel_metric1->append_child(param_power_kernel_metric1_k);

	GaussianKernel* gaussian_kernel=new GaussianKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	gaussian_kernel->print_modsel_params();

	ModelSelectionParameters* param_gaussian_kernel=
			new ModelSelectionParameters("kernel", gaussian_kernel);

	root->append_child(param_gaussian_kernel);

	ModelSelectionParameters* param_gaussian_kernel_width=
			new ModelSelectionParameters("log_width");
	param_gaussian_kernel_width->build_values(
	    0.0, 0.5 * std::log(2.0), R_LINEAR);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	ds_kernel->print_modsel_params();

	ModelSelectionParameters* param_ds_kernel=new ModelSelectionParameters("kernel",
			ds_kernel);

	root->append_child(param_ds_kernel);

	ModelSelectionParameters* param_ds_kernel_delta=
			new ModelSelectionParameters("delta");
	param_ds_kernel_delta->build_values(1.0, 2.0, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_delta);

	ModelSelectionParameters* param_ds_kernel_theta=
			new ModelSelectionParameters("theta");
	param_ds_kernel_theta->build_values(1.0, 2.0, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_theta);

	return root;
}

ModelSelectionParameters* build_sgobject_no_childs_tree()
{
	CPowerKernel* power_kernel=new CPowerKernel();
	ModelSelectionParameters* param_power_kernel=
			new ModelSelectionParameters("kernel", power_kernel);

	return param_power_kernel;
}

ModelSelectionParameters* build_leaf_node_tree()
{
	ModelSelectionParameters* c_1=new ModelSelectionParameters("C1");
	c_1->build_values(1.0, 1.0, R_EXP);

	return c_1;
}

ModelSelectionParameters* build_root_no_childs_tree()
{
	return new ModelSelectionParameters();
}

ModelSelectionParameters* build_root_value_childs_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c_1=new ModelSelectionParameters("C1");
	root->append_child(c_1);
	c_1->build_values(1.0, 1.0, R_EXP);

	ModelSelectionParameters* c_2=new ModelSelectionParameters("C2");
	root->append_child(c_2);
	c_2->build_values(1.0, 1.0, R_EXP);

	return root;
}

ModelSelectionParameters* build_root_sg_object_child_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	ModelSelectionParameters* param_power_kernel=
			new ModelSelectionParameters("kernel", power_kernel);

	root->append_child(param_power_kernel);

	return root;
}

ModelSelectionParameters* build_root_sg_object_child_value_child_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	ModelSelectionParameters* param_power_kernel=
			new ModelSelectionParameters("kernel", power_kernel);

	ModelSelectionParameters* c=new ModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(1.0, 1.0, R_EXP);

	root->append_child(param_power_kernel);

	return root;
}

void test_get_combinations(ModelSelectionParameters* tree)
{
	tree->print_tree();

	/* build combinations of parameter trees */
	DynamicObjectArray* combinations=tree->get_combinations();

	/* print and directly delete them all */
	SG_SPRINT("----------------------------------\n");
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* combination=(CParameterCombination*)
				combinations->get_element(i);
		combination->print_tree();
	}

}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	ModelSelectionParameters* tree;

	tree=build_root_no_childs_tree();
	test_get_combinations(tree);

	tree=build_leaf_node_tree();
	test_get_combinations(tree);

	tree=build_sgobject_no_childs_tree();
	test_get_combinations(tree);

	tree=build_root_value_childs_tree();
	test_get_combinations(tree);

	tree=build_root_sg_object_child_tree();
	test_get_combinations(tree);

	tree=build_root_sg_object_child_value_child_tree();
	test_get_combinations(tree);

	tree=build_complex_example_tree();
	test_get_combinations(tree);

	exit_shogun();

	return 0;
}

