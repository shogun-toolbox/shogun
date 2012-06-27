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
	SGMatrix<float64_t> matrix= SGMatrix<float64_t>(dim_vectors, num_vectors);

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
			
	SGMatrix<float64_t> matrix2= SGMatrix<float64_t>(dim_vectors, num_vectors);
	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
		matrix2[i]=i*sin(i)*.96;
	
	/* create training features */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
	features->set_feature_matrix(matrix);
	
	/* create testing features */
	CDenseFeatures<float64_t>* features2=new CDenseFeatures<float64_t> ();
	features2->set_feature_matrix(matrix2);
	
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

	CZeroMean* mean = new CZeroMean();
	CGaussianLikelihood* lik = new CGaussianLikelihood();
	lik->set_sigma(0.01);
	CExactInferenceMethod* inf = new CExactInferenceMethod(test_kernel, features, mean, labels, lik);
	SG_REF(inf);
	
	CGaussianProcessRegression* gp = new CGaussianProcessRegression(inf, features, labels);
	
	CModelSelectionParameters* root=new CModelSelectionParameters();
	
	CModelSelectionParameters* c2=new CModelSelectionParameters("Inference Method", inf);
	root->append_child(c2);
	
	CModelSelectionParameters* c3=new CModelSelectionParameters("Likelihood Model", lik);
	c2->append_child(c3); 

	CModelSelectionParameters* c1=new CModelSelectionParameters("sigma");
	c3->append_child(c1);
	c1->build_values(-10.0, 2.0, R_EXP);
	
	CModelSelectionParameters* c4=new CModelSelectionParameters("Kernel", test_kernel);
	c2->append_child(c4);

	CModelSelectionParameters* c5=new CModelSelectionParameters("width");
	c4->append_child(c5);
	c5->build_values(-10.0, 2.0, R_EXP);
	
	/* cross validation class for evaluation in model selection */
	SG_REF(gp);

	CGradientCriterion* crit = new CGradientCriterion();

	CGradientEvaluation* grad=new CGradientEvaluation(gp, features, labels,
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
	SGVector<float64_t> covariance = gp->getCovarianceVector(features2);
	CRegressionLabels* predictions = gp->apply_regression(features2);
	
	SGVector<float64_t>::display_vector(alpha.vector, alpha.vlen, "Alpha Vector");
	SGVector<float64_t>::display_vector(labe.vector, labe.vlen, "Labels");
	SGVector<float64_t>::display_vector(diagonal.vector, diagonal.vlen, "sW Matrix");
	SGVector<float64_t>::display_vector(covariance.vector, covariance.vlen, "Predicted Variances");
	SGVector<float64_t>::display_vector(predictions->get_labels().vector, predictions->get_labels().vlen, "Mean Predictions");
	SGMatrix<float64_t>::display_matrix(cholesky.matrix, cholesky.num_rows, cholesky.num_cols, "Cholesky Matrix L");
	SGMatrix<float64_t>::display_matrix(matrix.matrix, matrix.num_rows, matrix.num_cols, "Training Features");
	SGMatrix<float64_t>::display_matrix(matrix2.matrix, matrix2.num_rows, matrix2.num_cols, "Testing Features");
	
	/*free memory*/
	SG_UNREF(features);
	SG_UNREF(features2);
	SG_UNREF(predictions);
	SG_UNREF(labels);
	SG_UNREF(inf);
	SG_UNREF(gp);
	SG_UNREF(grad_search);
	SG_UNREF(best_combination);
	SG_UNREF(result);
	
	exit_shogun();

	return 0;
}
