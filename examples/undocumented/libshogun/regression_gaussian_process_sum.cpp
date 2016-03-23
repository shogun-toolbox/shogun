/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Jacob Walker
 */

#include <shogun/lib/config.h>
#if defined(HAVE_NLOPT)
#include <shogun/base/init.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/kernel/CombinedKernel.h>


using namespace shogun;

int32_t num_vectors=4;
int32_t dim_vectors=3;

void build_matrices(SGMatrix<float64_t>& test, SGMatrix<float64_t>& train,
		    CRegressionLabels* labels)
{
	/*Fill Matrices with random nonsense*/
	train[0] = -1;
	train[1] = -1;
	train[2] = -1;
	train[3] = 1;
	train[4] = 1;
	train[5] = 1;
	train[6] = -10;
	train[7] = -10;
	train[8] = -10;
	train[9] = 3;
	train[10] = 2;
	train[11] = 1;

	for (int32_t i=0; i<num_vectors*dim_vectors; i++)
	    test[i]=i*sin(i)*.96;

	/* create labels, two classes */
	for (index_t i=0; i<num_vectors; ++i)
	{
		if(i%2 == 0) labels->set_label(i, 1);
		else labels->set_label(i, -1);
	}
}

/* HEIKO FIXME
CModelSelectionParameters* build_tree(CInferenceMethod* inf,
				      CLikelihoodModel* lik, CCombinedKernel* kernel)
{
	CModelSelectionParameters* root=new CModelSelectionParameters();

	CModelSelectionParameters* c1 =
		new CModelSelectionParameters("inference_method", inf);

	root->append_child(c1);

        CModelSelectionParameters* c2 = new CModelSelectionParameters("scale");
        c1 ->append_child(c2);
        c2->build_values(0.99, 1.01, R_LINEAR);


	CModelSelectionParameters* c3 =
		new CModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3);

	CModelSelectionParameters* c4=new CModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(0.001, 1.0, R_LINEAR);

	CModelSelectionParameters* c5 =
		new CModelSelectionParameters("kernel", kernel);
	c1->append_child(c5);
	CList* list = kernel->get_list();
	CModelSelectionParameters* cc1 = new CModelSelectionParameters("kernel_list", list);
	c5->append_child(cc1);

	CListElement* first = NULL;
	CSGObject* k = list->get_first_element(first);
	SG_UNREF(k);
	SG_REF(first);

	CModelSelectionParameters* cc2 = new CModelSelectionParameters("first", first);
	cc1->append_child(cc2);

	CKernel* sub_kernel1 = kernel->get_kernel(0);
	CModelSelectionParameters* cc3 = new CModelSelectionParameters("data", sub_kernel1);
	cc2->append_child(cc3);
	SG_UNREF(sub_kernel1);

	CListElement* second = first;
	k = list->get_next_element(second);
	SG_UNREF(k);
	SG_REF(second);

	CModelSelectionParameters* cc4 = new CModelSelectionParameters("next", second);
	cc2->append_child(cc4);

	CKernel* sub_kernel2 = kernel->get_kernel(1);
	CModelSelectionParameters* cc5 = new CModelSelectionParameters("data", sub_kernel2);
	cc4->append_child(cc5);
	SG_UNREF(sub_kernel2);

	CListElement* third = second;
	k = list->get_next_element(third);
	SG_UNREF(k);
	SG_REF(third);

	CModelSelectionParameters* cc6 = new CModelSelectionParameters("next", third);
	cc4->append_child(cc6);

	CKernel* sub_kernel3 = kernel->get_kernel(2);
	CModelSelectionParameters* cc7 = new CModelSelectionParameters("data", sub_kernel3);
	cc6->append_child(cc7);
	SG_UNREF(sub_kernel3);

	CModelSelectionParameters* c6 =
		new CModelSelectionParameters("width");

	cc3->append_child(c6);
	c6->build_values(1.0, 4.0, R_LINEAR);

	CModelSelectionParameters* c66 =
		new CModelSelectionParameters("combined_kernel_weight");

	cc3->append_child(c66);
	c66->build_values(0.001, 1.0, R_LINEAR);

	CModelSelectionParameters* c7 =
		new CModelSelectionParameters("width");

	cc5->append_child(c7);
	c7->build_values(1.0, 4.0, R_LINEAR);

	CModelSelectionParameters* c77 =
		new CModelSelectionParameters("combined_kernel_weight");

	cc5->append_child(c77);
	c77->build_values(0.001, 1.0, R_LINEAR);

	CModelSelectionParameters* c8 =
		new CModelSelectionParameters("width");
	cc7->append_child(c8);
	c8->build_values(1.0, 4.0, R_LINEAR);

	CModelSelectionParameters* c88 =
		new CModelSelectionParameters("combined_kernel_weight");
	cc7->append_child(c88);
	c88->build_values(0.001, 1.0, R_LINEAR);

	SG_UNREF(list);

	return root;
}
*/

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	/* create some data and labels */
	SGMatrix<float64_t> matrix =
		 SGMatrix<float64_t>(dim_vectors, num_vectors);

	SGMatrix<float64_t> matrix2 =
		SGMatrix<float64_t>(dim_vectors, num_vectors);

	CRegressionLabels* labels=new CRegressionLabels(num_vectors);

	build_matrices(matrix2, matrix, labels);

	/* create training features */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t> ();
	features->set_feature_matrix(matrix);

	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);


	CCombinedKernel* test_kernel = new CCombinedKernel();
	CGaussianKernel* sub_kernel1 = new CGaussianKernel(10, 2);
	CGaussianKernel* sub_kernel2 = new CGaussianKernel(10, 2);
	CGaussianKernel* sub_kernel3 = new CGaussianKernel(10, 2);

	test_kernel->append_kernel(sub_kernel1);
	test_kernel->append_kernel(sub_kernel2);
	test_kernel->append_kernel(sub_kernel3);

	SG_REF(comb_features);
	SG_REF(labels);

	/*Allocate our Mean Function*/
	CZeroMean* mean = new CZeroMean();

	/*Allocate our Likelihood Model*/
	CGaussianLikelihood* lik = new CGaussianLikelihood();

	/*Allocate our inference method*/
	CExactInferenceMethod* inf =
			new CExactInferenceMethod(test_kernel,
						  comb_features, mean, labels, lik);

	SG_REF(inf);

	/*Finally use these to allocate the Gaussian Process Object*/
	CGaussianProcessRegression* gp =
		new CGaussianProcessRegression(inf);

	SG_REF(gp);

	//CModelSelectionParameters* root = build_tree(inf, lik, test_kernel);
	//
	///*Criterion for gradient search*/
	//CGradientCriterion* crit = new CGradientCriterion();

	///*This will evaluate our inference method for its derivatives*/
	//CGradientEvaluation* grad=new CGradientEvaluation(gp, comb_features, labels,
	//		crit);

	//grad->set_function(inf);

	//gp->print_modsel_params();

	//root->print_tree();

	///* handles all of the above structures in memory */
	//CGradientModelSelection* grad_search=new CGradientModelSelection(
	//		root, grad);

	///* set autolocking to false to get rid of warnings */
	//grad->set_autolock(false);

	///*Search for best parameters*/
	//CParameterCombination* best_combination=grad_search->select_model(true);

	///*Output all the results and information*/
	//if (best_combination)
	//{
	//	SG_SPRINT("best parameter(s):\n");
	//	best_combination->print_tree();

	//	best_combination->apply_to_machine(gp);
	//}

	//CGradientResult* result=(CGradientResult*)grad->evaluate();

	//if(result->get_result_type() != GRADIENTEVALUATION_RESULT)
	//	SG_SERROR("Evaluation result not a GradientEvaluationResult!");

	//result->print_result();

	//SGVector<float64_t> alpha = inf->get_alpha();
	//SGVector<float64_t> labe = labels->get_labels();
	//SGVector<float64_t> diagonal = inf->get_diagonal_vector();
	//SGMatrix<float64_t> cholesky = inf->get_cholesky();
	//gp->set_return_type(CGaussianProcessRegression::GP_RETURN_COV);

	//CRegressionLabels* covariance = gp->apply_regression(comb_features);

	//gp->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	//
	//CRegressionLabels* predictions = gp->apply_regression();

	//alpha.display_vector("Alpha Vector");
	//labe.display_vector("Labels");
	//diagonal.display_vector("sW Matrix");
	//covariance->get_labels().display_vector("Predicted Variances");
	//predictions->get_labels().display_vector("Mean Predictions");
	//cholesky.display_matrix("Cholesky Matrix L");
	//matrix.display_matrix("Training Features");
	//matrix2.display_matrix("Testing Features");

	///*free memory*/
	//SG_UNREF(predictions);
	//SG_UNREF(covariance);
	SG_UNREF(labels);
	SG_UNREF(comb_features);
	SG_UNREF(inf);
	SG_UNREF(gp);
	//SG_UNREF(grad_search);
	//SG_UNREF(best_combination);
	//SG_UNREF(result);

	exit_shogun();

	return 0;

}
#else // HAVE_NLOPT
int main(int argc, char **argv)
{
	return 0;
}
#endif // HAVE_NLOPT
