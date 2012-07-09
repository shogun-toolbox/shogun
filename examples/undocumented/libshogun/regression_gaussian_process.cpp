/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Jacob Walker
 */

#include <shogun/base/init.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/kernel/CombinedKernel.h>


using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}


int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	int32_t num_vectors=4;
	int32_t dim_vectors=3;

	/* create some data and labels */
	SGMatrix<float64_t> matrix =
		 SGMatrix<float64_t>(dim_vectors, num_vectors);

	matrix[0] = -1;
	matrix[1] = -1;
	matrix[2] = -1;
	matrix[3] = 1;
	matrix[4] = 1;
	matrix[5] = 1;
	matrix[6] = -10;
	matrix[7] = -10;
	matrix[8] = -10;
	matrix[9] = 3;
	matrix[10] = 2;
	matrix[11] = 1;
			
	SGMatrix<float64_t> matrix2 = 
		SGMatrix<float64_t>(dim_vectors, num_vectors);

	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
		matrix2[i]=i*sin(i)*.96;
	
	/* create training features */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
	features->set_feature_matrix(matrix);
	
	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	/* create testing features */
	CDenseFeatures<float64_t>* features2=new CDenseFeatures<float64_t> ();
	features2->set_feature_matrix(matrix2);
	
/*	CCombinedFeatures* comb_features2=new CCombinedFeatures();
	comb_features2->append_feature_obj(features2);
	comb_features2->append_feature_obj(features2);
	comb_features2->append_feature_obj(features2);*/

	CGaussianKernel* sub_kernel1 = new CGaussianKernel(10, 2);
	CGaussianKernel* sub_kernel2 = new CGaussianKernel(10, 2);
	CGaussianKernel* sub_kernel3 = new CGaussianKernel(10, 2);
	
	CCombinedKernel* kernel1=new CCombinedKernel();
	kernel1->append_kernel(sub_kernel1);
	kernel1->append_kernel(sub_kernel2);
	kernel1->append_kernel(sub_kernel3);


	SG_REF(features);
	SG_REF(features2);
	
	CRegressionLabels* labels=new CRegressionLabels(num_vectors);

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
	{
		if(i%2 == 0) labels->set_label(i, 1);
		else labels->set_label(i, -1);
	}
	
	SG_REF(labels);
	CGaussianKernel* test_kernel = new CGaussianKernel(10, 2);
	
	test_kernel->init(features, features);
	 
	kernel1->init(comb_features, comb_features);

	CZeroMean* mean = new CZeroMean();
	CGaussianLikelihood* lik = new CGaussianLikelihood();
	lik->set_sigma(0.01);
	
	CExactInferenceMethod* inf =
		 new CExactInferenceMethod(kernel1, comb_features, mean, labels, lik);
	
	SG_REF(inf);
	
	CGaussianProcessRegression* gp = 
		new CGaussianProcessRegression(inf, comb_features, labels);

	CModelSelectionParameters* root=new CModelSelectionParameters();
	
	CModelSelectionParameters* c1 = 
		new CModelSelectionParameters("inference_method", inf);
	root->append_child(c1);

        CModelSelectionParameters* c2 = new CModelSelectionParameters("scale");
        c1 ->append_child(c2);
        c2->build_values(0.01, 4.0, R_LINEAR);

	
	CModelSelectionParameters* c3 = 
		new CModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3); 

	CModelSelectionParameters* c4=new CModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(0.001, 4.0, R_LINEAR);
	
	CModelSelectionParameters* c5 = 
		new CModelSelectionParameters("kernel", kernel1);
	c1->append_child(c5);
	
	CModelSelectionParameters* cc1 = new CModelSelectionParameters("kernel_list", kernel1->get_list());
	c5->append_child(cc1);
	
	CModelSelectionParameters* cc2 = new CModelSelectionParameters("first", kernel1->get_list()->first);
	cc1->append_child(cc2);
	
	CListElement* cl = (CListElement*)kernel1->get_list()->first;
	
	CModelSelectionParameters* cc3 = new CModelSelectionParameters("data", sub_kernel1);
	cc2->append_child(cc3);

	CModelSelectionParameters* cc4 = new CModelSelectionParameters("next", cl->next);
	cc2->append_child(cc4);
	
	CModelSelectionParameters* cc5 = new CModelSelectionParameters("data", sub_kernel2);
	cc4->append_child(cc5);
	
	CModelSelectionParameters* cc6 = new CModelSelectionParameters("next", cl->next->next);
	cc4->append_child(cc6);
	
	CModelSelectionParameters* cc7 = new CModelSelectionParameters("data", sub_kernel3);
	cc6->append_child(cc7);
	
	CModelSelectionParameters* c6 = 
		new CModelSelectionParameters("width");
	cc3->append_child(c6);
	c6->build_values(0.001, 4.0, R_LINEAR);
	
		CModelSelectionParameters* c66 = 
		new CModelSelectionParameters("combined_kernel_weight");
	cc3->append_child(c66);
	c66->build_values(0.001, 4.0, R_LINEAR);
	
		CModelSelectionParameters* c7 = 
		new CModelSelectionParameters("width");
	cc5->append_child(c7);
	c7->build_values(0.001, 4.0, R_LINEAR);
	
			CModelSelectionParameters* c77 = 
		new CModelSelectionParameters("combined_kernel_weight");
	cc5->append_child(c77);
	c77->build_values(0.001, 4.0, R_LINEAR);
	
		CModelSelectionParameters* c8 = 
		new CModelSelectionParameters("width");
	cc7->append_child(c8);
	c8->build_values(0.001, 4.0, R_LINEAR);
	
			CModelSelectionParameters* c88 = 
		new CModelSelectionParameters("combined_kernel_weight");
	cc7->append_child(c88);
	c88->build_values(0.001, 4.0, R_LINEAR);
	
	
	/* cross validation class for evaluation in model selection */
	SG_REF(gp);

	CGradientCriterion* crit = new CGradientCriterion();

	CGradientEvaluation* grad=new CGradientEvaluation(gp, comb_features, labels,
			crit);
	
	grad->set_function(inf);
	
	gp->print_modsel_params();
	
	root->print_tree();
	
	/* handles all of the above structures in memory */
	CGradientModelSelection* grad_search=new CGradientModelSelection(
	  root, grad);

	/* set autolocking to false to get rid of warnings */
	grad->set_autolock(false);

	CParameterCombination* best_combination=grad_search->select_model(true);

	if (best_combination)
	{
		SG_SPRINT("best parameter(s):\n");
		best_combination->print_tree();

		best_combination->apply_to_machine(gp);
	}

	CGradientResult* result=(CGradientResult*)grad->evaluate();

	if(result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_SERROR("Evaluation result not a GradientEvaluationResult!");

	result->print_result();

	SGVector<float64_t> alpha = inf->get_alpha();
	SGVector<float64_t> labe = labels->get_labels();
	SGVector<float64_t> diagonal = inf->get_diagonal_vector();
	SGMatrix<float64_t> cholesky = inf->get_cholesky();
	gp->set_return_type(CGaussianProcessRegression::GP_RETURN_COV);
	
	CRegressionLabels* covariance = gp->apply_regression(comb_features);
	
	gp->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions = gp->apply_regression();
	
	alpha.display_vector("Alpha Vector");
	labe.display_vector("Labels");
	diagonal.display_vector("sW Matrix");
	covariance->get_labels().display_vector("Predicted Variances");
	predictions->get_labels().display_vector("Mean Predictions");
	cholesky.display_matrix("Cholesky Matrix L");
	matrix.display_matrix("Training Features");
	matrix2.display_matrix("Testing Features");
	
	/*free memory*/
	SG_UNREF(features);
	SG_UNREF(features2);
	SG_UNREF(predictions);
	SG_UNREF(covariance);
	SG_UNREF(labels);
	SG_UNREF(inf);
	SG_UNREF(gp);
	SG_UNREF(grad_search);
	SG_UNREF(best_combination);
	SG_UNREF(result);
	
	exit_shogun();

	return 0;
}
