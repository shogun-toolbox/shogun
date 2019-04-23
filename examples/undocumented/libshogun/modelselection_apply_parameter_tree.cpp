/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Roman Votyakov, 
 *          Sergey Lisitsyn, Wu Lin
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

ModelSelectionParameters* create_param_tree()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c=new ModelSelectionParameters("C1");
	root->append_child(c);
	c->build_values(1.0, 2.0, R_EXP);

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

	return root;
}

void apply_parameter_tree(DynamicObjectArray* combinations)
{
	/* create some data */
	SGMatrix<float64_t> matrix(2,3);
	for (index_t i=0; i<6; i++)
		matrix.matrix[i]=i;

	/* create three 2-dimensional vectors
	 * to avoid deleting these, REF now and UNREF when finished */
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);

	/* create three labels, will be handed to svm and automaticall deleted */
	BinaryLabels* labels=new BinaryLabels(3);
	labels->set_label(0, -1);
	labels->set_label(1, +1);
	labels->set_label(2, -1);

	/* create libsvm with C=10 and train */
	CLibSVM* svm=new CLibSVM();
	svm->set_labels(labels);

	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		SG_SPRINT("applying:\n");
		CParameterCombination* current_combination=(CParameterCombination*)
				combinations->get_element(i);
		current_combination->print_tree();
		Parameter* current_parameters=svm->m_parameters;
		current_combination->apply_to_modsel_parameter(current_parameters);

		Kernel* kernel=svm->get_kernel();
		kernel->init(features, features);

		svm->train();

		/* classify on training examples */
		for (index_t j=0; j<3; j++)
			SG_SPRINT("output[%d]=%f\n", j, svm->apply_one(j));

		kernel->cleanup();

		SG_SPRINT("----------------\n\n");
	}

	/* free up memory */
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* create example tree */
	ModelSelectionParameters* tree=create_param_tree();
	tree->print_tree();
	SG_SPRINT("----------------------------------\n");

	/* build combinations of parameter trees */
	DynamicObjectArray* combinations=tree->get_combinations();

	apply_parameter_tree(combinations);

	/* print and directly delete them all */
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* combination=(CParameterCombination*)
				combinations->get_element(i);
	}


	/* delete example tree (after processing of combinations because SGObject
	 * else) */

	exit_shogun();

	return 0;
}
