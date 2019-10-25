/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Roman Votyakov, Heiko Strathmann, Soumyajit De, Sergey Lisitsyn, 
 *          Wu Lin
 */

#include <shogun/base/ShogunEnv.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/modelselection/ModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/MulticlassLibSVM.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

ModelSelectionParameters* build_param_tree(Kernel* kernel)
{
	ModelSelectionParameters * root=new ModelSelectionParameters();
	ModelSelectionParameters * c=new ModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(-1.0, 1.0, R_EXP);

	ModelSelectionParameters * params_kernel=new ModelSelectionParameters("kernel", kernel);
	root->append_child(params_kernel);
	ModelSelectionParameters * params_kernel_width=new ModelSelectionParameters("log_width");
	params_kernel_width->build_values(-std::log(2.0), 0.0, R_LINEAR);
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
			feat(i, j)=Math::randn_double();

		/* make sure classes are (alomst) linearly seperable against each other */
		feat(lab[j],j)+=distance;
	}

	/* shogun representation of above data */
	DenseFeatures<float64_t> * cfeatures=new DenseFeatures<float64_t>(feat);
	MulticlassLabels * clabels=new MulticlassLabels(lab);

	float64_t sigma=2;
	GaussianKernel* kernel=new GaussianKernel(10, sigma);

	const float C=10.;
	CMulticlassLibSVM* cmachine=new CMulticlassLibSVM(C, kernel, clabels);

	MulticlassAccuracy * eval_crit=new MulticlassAccuracy();

	/* k-fold stratified x-validation */
	index_t k=3;
	StratifiedCrossValidationSplitting * splitting=
			new StratifiedCrossValidationSplitting(clabels, k);

	CrossValidation * cross=new CrossValidation(cmachine, cfeatures, clabels,
			splitting, eval_crit);
	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.05);

	/* create peramters for model selection */
	ModelSelectionParameters* root=build_param_tree(kernel);

	CGridSearchModelSelection * model_selection=new CGridSearchModelSelection(
			cross, root);
	bool print_state=true;
	CParameterCombination * params=model_selection->select_model(print_state);
	SG_SPRINT("best combination\n");
	params->print_tree();

	/* clean up memory */
}

int main(int argc, char **argv)
{
	env()->io()->set_loglevel(MSG_DEBUG);

	test();

	return 0;
}
