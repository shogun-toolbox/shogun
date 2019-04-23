/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Jacob Walker, Chiyuan Zhang, 
 *          Roman Votyakov, Pan Deng, Wu Lin
 */

#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>

using namespace shogun;

void test_tree(ModelSelectionParameters* tree)
{
	SG_SPRINT("\n\ntree to process:\n");
	tree->print_tree();

	/* build combinations of parameter trees */
	DynamicObjectArray* combinations=tree->get_combinations();

	/* print and directly delete them all */
	SG_SPRINT("----------------------------------\n");
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* combination=
				(CParameterCombination*)combinations->get_element(i);
		combination->print_tree();
	}

}

ModelSelectionParameters* create_param_tree_1()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c=new ModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(1, 2, R_EXP);

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	ModelSelectionParameters* param_power_kernel=new ModelSelectionParameters(
			"kernel", power_kernel);

	root->append_child(param_power_kernel);

	ModelSelectionParameters* param_power_kernel_degree=
			new ModelSelectionParameters("degree");
	param_power_kernel_degree->build_values(1, 2, R_EXP);
	param_power_kernel->append_child(param_power_kernel_degree);

	CMinkowskiMetric* m_metric=new CMinkowskiMetric(10);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	m_metric->print_modsel_params();

	ModelSelectionParameters* param_power_kernel_metrikernel_width_sigma_param=
			new ModelSelectionParameters("distance", m_metric);

	param_power_kernel->append_child(
			param_power_kernel_metrikernel_width_sigma_param);

	ModelSelectionParameters* param_power_kernel_metrikernel_width_sigma_param_k=
			new ModelSelectionParameters("k");
	param_power_kernel_metrikernel_width_sigma_param_k->build_values(1, 2,
			R_LINEAR);
	param_power_kernel_metrikernel_width_sigma_param->append_child(
			param_power_kernel_metrikernel_width_sigma_param_k);

	GaussianKernel* gaussian_kernel=new GaussianKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	gaussian_kernel->print_modsel_params();

	ModelSelectionParameters* param_gaussian_kernel=
			new ModelSelectionParameters("kernel", gaussian_kernel);

	root->append_child(param_gaussian_kernel);

	ModelSelectionParameters* param_gaussian_kernel_width=
			new ModelSelectionParameters("log_width");
	param_gaussian_kernel_width->build_values(0.0, 0.5 * std::log(2), R_LINEAR);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	ds_kernel->print_modsel_params();

	ModelSelectionParameters* param_ds_kernel=new ModelSelectionParameters(
			"kernel", ds_kernel);

	root->append_child(param_ds_kernel);

	ModelSelectionParameters* param_ds_kernel_delta=
			new ModelSelectionParameters("delta");
	param_ds_kernel_delta->build_values(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_delta);

	ModelSelectionParameters* param_ds_kernel_theta=
			new ModelSelectionParameters("theta");
	param_ds_kernel_theta->build_values(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_theta);

	return root;
}

ModelSelectionParameters* create_param_tree_2()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();
	ModelSelectionParameters* param_power_kernel=new ModelSelectionParameters(
			"kernel", power_kernel);
	root->append_child(param_power_kernel);

	CMinkowskiMetric* metric=new CMinkowskiMetric();
	ModelSelectionParameters* param_power_kernel_metric=
			new ModelSelectionParameters("distance", metric);
	param_power_kernel->append_child(param_power_kernel_metric);

	ModelSelectionParameters* param_metric_k=new ModelSelectionParameters(
			"k");
	param_metric_k->build_values(2, 3, R_LINEAR);
	param_power_kernel_metric->append_child(param_metric_k);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();
	ModelSelectionParameters* param_ds_kernel=new ModelSelectionParameters(
			"kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	return root;
}

ModelSelectionParameters* create_param_tree_3()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();
	ModelSelectionParameters* param_power_kernel=new ModelSelectionParameters(
			"kernel", power_kernel);
	root->append_child(param_power_kernel);

	CMinkowskiMetric* metric=new CMinkowskiMetric();
	ModelSelectionParameters* param_power_kernel_metric=
			new ModelSelectionParameters("distance", metric);
	param_power_kernel->append_child(param_power_kernel_metric);

	EuclideanDistance* euclidean=new EuclideanDistance();
	ModelSelectionParameters* param_power_kernel_distance=
			new ModelSelectionParameters("distance", euclidean);
	param_power_kernel->append_child(param_power_kernel_distance);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();
	ModelSelectionParameters* param_ds_kernel=new ModelSelectionParameters(
			"kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	return root;
}

ModelSelectionParameters* create_param_tree_4a()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>();
	RegressionLabels* labels=new RegressionLabels();
	GaussianKernel* gaussian_kernel=new GaussianKernel(10, 2);
	CPowerKernel* power_kernel=new CPowerKernel();

	ZeroMean* mean=new ZeroMean();
	GaussianLikelihood* lik=new GaussianLikelihood();
	ExactInferenceMethod* inf=new ExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	CLibSVM* svm=new CLibSVM();
	CPowerKernel* power_kernel_svm=new CPowerKernel();
	GaussianKernel* gaussian_kernel_svm=new GaussianKernel(10, 2);

	ModelSelectionParameters* param_inf=new ModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	ModelSelectionParameters* param_inf_gaussian=new ModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	ModelSelectionParameters* param_inf_kernel_1=new ModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	ModelSelectionParameters* param_inf_kernel_2=new ModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_2);



	ModelSelectionParameters* param_svm=new ModelSelectionParameters(
			"SVM", svm);
	root->append_child(param_svm);

	ModelSelectionParameters* param_svm_kernel_1=new ModelSelectionParameters(
			"kernel", power_kernel_svm);
	param_svm->append_child(param_svm_kernel_1);

	ModelSelectionParameters* param_svm_kernel_2=new ModelSelectionParameters(
				"kernel", gaussian_kernel_svm);
	param_svm->append_child(param_svm_kernel_2);

	return root;
}

ModelSelectionParameters* create_param_tree_4b()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>();
	RegressionLabels* labels=new RegressionLabels();
	GaussianKernel* gaussian_kernel=new GaussianKernel(10, 2);
	CPowerKernel* power_kernel=new CPowerKernel();

	ZeroMean* mean=new ZeroMean();
	GaussianLikelihood* lik=new GaussianLikelihood();
	ExactInferenceMethod* inf=new ExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	CLibSVM* svm=new CLibSVM();
	CPowerKernel* power_kernel_svm=new CPowerKernel();
	GaussianKernel* gaussian_kernel_svm=new GaussianKernel(10, 2);

	ModelSelectionParameters* param_c=new ModelSelectionParameters("C1");
	root->append_child(param_c);
	param_c->build_values(1,2,R_EXP);

	ModelSelectionParameters* param_inf=new ModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	ModelSelectionParameters* param_inf_gaussian=new ModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	ModelSelectionParameters* param_inf_kernel_1=new ModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	ModelSelectionParameters* param_inf_kernel_2=new ModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_2);



	ModelSelectionParameters* param_svm=new ModelSelectionParameters(
			"SVM", svm);
	root->append_child(param_svm);

	ModelSelectionParameters* param_svm_kernel_1=new ModelSelectionParameters(
			"kernel", power_kernel_svm);
	param_svm->append_child(param_svm_kernel_1);

	ModelSelectionParameters* param_svm_kernel_2=new ModelSelectionParameters(
				"kernel", gaussian_kernel_svm);
	param_svm->append_child(param_svm_kernel_2);

	return root;
}

ModelSelectionParameters* create_param_tree_5()
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>();
	RegressionLabels* labels=new RegressionLabels();
	GaussianKernel* gaussian_kernel=new GaussianKernel(10, 2);
	LinearKernel* linear_kernel=new LinearKernel();
	CPowerKernel* power_kernel=new CPowerKernel();

	ZeroMean* mean=new ZeroMean();
	GaussianLikelihood* lik=new GaussianLikelihood();
	ExactInferenceMethod* inf=new ExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	ModelSelectionParameters* param_inf=new ModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	ModelSelectionParameters* param_inf_gaussian=new ModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	ModelSelectionParameters* param_inf_gaussian_sigma=
			new ModelSelectionParameters("log_sigma");
	param_inf_gaussian->append_child(param_inf_gaussian_sigma);
	param_inf_gaussian_sigma->build_values(
	    2.0 * std::log(2.0), 3.0 * std::log(2.0), R_LINEAR);

	ModelSelectionParameters* param_inf_kernel_1=new ModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	ModelSelectionParameters* param_inf_kernel_width=
			new ModelSelectionParameters("log_width");
	param_inf_kernel_1->append_child(param_inf_kernel_width);
	param_inf_kernel_width->build_values(0.0, 0.5 * std::log(2.0), R_LINEAR);

	ModelSelectionParameters* param_inf_kernel_2=new ModelSelectionParameters(
			"kernel", linear_kernel);
	param_inf->append_child(param_inf_kernel_2);

	ModelSelectionParameters* param_inf_kernel_3=new ModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_3);

	return root;
}

int main(int argc, char **argv)
{
//	env()->io()->set_loglevel(MSG_DEBUG);

	ModelSelectionParameters* tree=NULL;

	tree=create_param_tree_1();
	test_tree(tree);

	tree=create_param_tree_2();
	test_tree(tree);

	tree=create_param_tree_3();
	test_tree(tree);

	tree=create_param_tree_4a();
	test_tree(tree);

	tree=create_param_tree_4b();
	test_tree(tree);

	tree=create_param_tree_5();
	test_tree(tree);

	return 0;
}
