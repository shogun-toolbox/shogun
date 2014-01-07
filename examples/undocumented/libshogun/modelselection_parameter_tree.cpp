/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>
#include <classifier/svm/LibSVM.h>
#include <kernel/GaussianKernel.h>
#include <kernel/PowerKernel.h>
#include <kernel/LinearKernel.h>
#include <distance/MinkowskiMetric.h>
#include <distance/EuclideanDistance.h>
#include <kernel/string/DistantSegmentsKernel.h>
#include <machine/gp/ExactInferenceMethod.h>
#include <machine/gp/GaussianLikelihood.h>
#include <machine/gp/ZeroMean.h>
#include <regression/GaussianProcessRegression.h>
#include <modelselection/ModelSelectionParameters.h>
#include <modelselection/ParameterCombination.h>

using namespace shogun;

void test_tree(CModelSelectionParameters* tree)
{
	SG_SPRINT("\n\ntree to process:\n");
	tree->print_tree();

	/* build combinations of parameter trees */
	CDynamicObjectArray* combinations=tree->get_combinations();

	/* print and directly delete them all */
	SG_SPRINT("----------------------------------\n");
	for (index_t i=0; i<combinations->get_num_elements(); ++i)
	{
		CParameterCombination* combination=
				(CParameterCombination*)combinations->get_element(i);
		combination->print_tree();
		SG_UNREF(combination);
	}

	SG_UNREF(combinations);
}

CModelSelectionParameters* create_param_tree_1()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* c=new CModelSelectionParameters("C");
	root->append_child(c);
	c->build_values(1, 2, R_EXP);

	CPowerKernel* power_kernel=new CPowerKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	power_kernel->print_modsel_params();

	CModelSelectionParameters* param_power_kernel=new CModelSelectionParameters(
			"kernel", power_kernel);

	root->append_child(param_power_kernel);

	CModelSelectionParameters* param_power_kernel_degree=
			new CModelSelectionParameters("degree");
	param_power_kernel_degree->build_values(1, 2, R_EXP);
	param_power_kernel->append_child(param_power_kernel_degree);

	CMinkowskiMetric* m_metric=new CMinkowskiMetric(10);

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	m_metric->print_modsel_params();

	CModelSelectionParameters* param_power_kernel_metrikernel_width_sigma_param=
			new CModelSelectionParameters("distance", m_metric);

	param_power_kernel->append_child(
			param_power_kernel_metrikernel_width_sigma_param);

	CModelSelectionParameters* param_power_kernel_metrikernel_width_sigma_param_k=
			new CModelSelectionParameters("k");
	param_power_kernel_metrikernel_width_sigma_param_k->build_values(1, 2,
			R_LINEAR);
	param_power_kernel_metrikernel_width_sigma_param->append_child(
			param_power_kernel_metrikernel_width_sigma_param_k);

	CGaussianKernel* gaussian_kernel=new CGaussianKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	gaussian_kernel->print_modsel_params();

	CModelSelectionParameters* param_gaussian_kernel=
			new CModelSelectionParameters("kernel", gaussian_kernel);

	root->append_child(param_gaussian_kernel);

	CModelSelectionParameters* param_gaussian_kernel_width=
			new CModelSelectionParameters("width");
	param_gaussian_kernel_width->build_values(1, 2, R_EXP);
	param_gaussian_kernel->append_child(param_gaussian_kernel_width);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();

	/* print all parameter available for modelselection
	 * Dont worry if yours is not included, simply write to the mailing list */
	ds_kernel->print_modsel_params();

	CModelSelectionParameters* param_ds_kernel=new CModelSelectionParameters(
			"kernel", ds_kernel);

	root->append_child(param_ds_kernel);

	CModelSelectionParameters* param_ds_kernel_delta=
			new CModelSelectionParameters("delta");
	param_ds_kernel_delta->build_values(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_delta);

	CModelSelectionParameters* param_ds_kernel_theta=
			new CModelSelectionParameters("theta");
	param_ds_kernel_theta->build_values(1, 2, R_EXP);
	param_ds_kernel->append_child(param_ds_kernel_theta);

	return root;
}

CModelSelectionParameters* create_param_tree_2()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();
	CModelSelectionParameters* param_power_kernel=new CModelSelectionParameters(
			"kernel", power_kernel);
	root->append_child(param_power_kernel);

	CMinkowskiMetric* metric=new CMinkowskiMetric();
	CModelSelectionParameters* param_power_kernel_metric=
			new CModelSelectionParameters("distance", metric);
	param_power_kernel->append_child(param_power_kernel_metric);

	CModelSelectionParameters* param_metric_k=new CModelSelectionParameters(
			"k");
	param_metric_k->build_values(2, 3, R_LINEAR);
	param_power_kernel_metric->append_child(param_metric_k);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();
	CModelSelectionParameters* param_ds_kernel=new CModelSelectionParameters(
			"kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	return root;
}

CModelSelectionParameters* create_param_tree_3()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CPowerKernel* power_kernel=new CPowerKernel();
	CModelSelectionParameters* param_power_kernel=new CModelSelectionParameters(
			"kernel", power_kernel);
	root->append_child(param_power_kernel);

	CMinkowskiMetric* metric=new CMinkowskiMetric();
	CModelSelectionParameters* param_power_kernel_metric=
			new CModelSelectionParameters("distance", metric);
	param_power_kernel->append_child(param_power_kernel_metric);

	CEuclideanDistance* euclidean=new CEuclideanDistance();
	CModelSelectionParameters* param_power_kernel_distance=
			new CModelSelectionParameters("distance", euclidean);
	param_power_kernel->append_child(param_power_kernel_distance);

	CDistantSegmentsKernel* ds_kernel=new CDistantSegmentsKernel();
	CModelSelectionParameters* param_ds_kernel=new CModelSelectionParameters(
			"kernel", ds_kernel);
	root->append_child(param_ds_kernel);

	return root;
}

#ifdef HAVE_EIGEN3
CModelSelectionParameters* create_param_tree_4a()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	CRegressionLabels* labels=new CRegressionLabels();
	CGaussianKernel* gaussian_kernel=new CGaussianKernel(10, 2);
	CPowerKernel* power_kernel=new CPowerKernel();

	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	CExactInferenceMethod* inf=new CExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	CLibSVM* svm=new CLibSVM();
	CPowerKernel* power_kernel_svm=new CPowerKernel();
	CGaussianKernel* gaussian_kernel_svm=new CGaussianKernel(10, 2);

	CModelSelectionParameters* param_inf=new CModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	CModelSelectionParameters* param_inf_gaussian=new CModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	CModelSelectionParameters* param_inf_kernel_1=new CModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	CModelSelectionParameters* param_inf_kernel_2=new CModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_2);



	CModelSelectionParameters* param_svm=new CModelSelectionParameters(
			"SVM", svm);
	root->append_child(param_svm);

	CModelSelectionParameters* param_svm_kernel_1=new CModelSelectionParameters(
			"kernel", power_kernel_svm);
	param_svm->append_child(param_svm_kernel_1);

	CModelSelectionParameters* param_svm_kernel_2=new CModelSelectionParameters(
				"kernel", gaussian_kernel_svm);
	param_svm->append_child(param_svm_kernel_2);

	return root;
}

CModelSelectionParameters* create_param_tree_4b()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	CRegressionLabels* labels=new CRegressionLabels();
	CGaussianKernel* gaussian_kernel=new CGaussianKernel(10, 2);
	CPowerKernel* power_kernel=new CPowerKernel();

	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	CExactInferenceMethod* inf=new CExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	CLibSVM* svm=new CLibSVM();
	CPowerKernel* power_kernel_svm=new CPowerKernel();
	CGaussianKernel* gaussian_kernel_svm=new CGaussianKernel(10, 2);

	CModelSelectionParameters* param_c=new CModelSelectionParameters("C1");
	root->append_child(param_c);
	param_c->build_values(1,2,R_EXP);

	CModelSelectionParameters* param_inf=new CModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	CModelSelectionParameters* param_inf_gaussian=new CModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	CModelSelectionParameters* param_inf_kernel_1=new CModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	CModelSelectionParameters* param_inf_kernel_2=new CModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_2);



	CModelSelectionParameters* param_svm=new CModelSelectionParameters(
			"SVM", svm);
	root->append_child(param_svm);

	CModelSelectionParameters* param_svm_kernel_1=new CModelSelectionParameters(
			"kernel", power_kernel_svm);
	param_svm->append_child(param_svm_kernel_1);

	CModelSelectionParameters* param_svm_kernel_2=new CModelSelectionParameters(
				"kernel", gaussian_kernel_svm);
	param_svm->append_child(param_svm_kernel_2);

	return root;
}

CModelSelectionParameters* create_param_tree_5()
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	CRegressionLabels* labels=new CRegressionLabels();
	CGaussianKernel* gaussian_kernel=new CGaussianKernel(10, 2);
	CLinearKernel* linear_kernel=new CLinearKernel();
	CPowerKernel* power_kernel=new CPowerKernel();

	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	CExactInferenceMethod* inf=new CExactInferenceMethod(gaussian_kernel, features,
			mean, labels, lik);

	CModelSelectionParameters* param_inf=new CModelSelectionParameters(
			"inference_method", inf);
	root->append_child(param_inf);

	CModelSelectionParameters* param_inf_gaussian=new CModelSelectionParameters(
			"likelihood_model", lik);
	param_inf->append_child(param_inf_gaussian);

	CModelSelectionParameters* param_inf_gaussian_sigma=
			new CModelSelectionParameters("sigma");
	param_inf_gaussian->append_child(param_inf_gaussian_sigma);
	param_inf_gaussian_sigma->build_values(2.0, 3.0, R_EXP);

	CModelSelectionParameters* param_inf_kernel_1=new CModelSelectionParameters(
			"kernel", gaussian_kernel);
	param_inf->append_child(param_inf_kernel_1);

	CModelSelectionParameters* param_inf_kernel_width=
			new CModelSelectionParameters("width");
	param_inf_kernel_1->append_child(param_inf_kernel_width);
	param_inf_kernel_width->build_values(1.0, 2.0, R_EXP);

	CModelSelectionParameters* param_inf_kernel_2=new CModelSelectionParameters(
			"kernel", linear_kernel);
	param_inf->append_child(param_inf_kernel_2);

	CModelSelectionParameters* param_inf_kernel_3=new CModelSelectionParameters(
			"kernel", power_kernel);
	param_inf->append_child(param_inf_kernel_3);

	return root;
}
#endif

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	CModelSelectionParameters* tree=NULL;

	tree=create_param_tree_1();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);

	tree=create_param_tree_2();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);

	tree=create_param_tree_3();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);

#ifdef HAVE_EIGEN3
	tree=create_param_tree_4a();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);

	tree=create_param_tree_4b();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);

	tree=create_param_tree_5();
	SG_REF(tree);
	test_tree(tree);
	SG_UNREF(tree);
#endif

	exit_shogun();

	return 0;
}
