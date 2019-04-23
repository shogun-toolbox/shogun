/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Jacob Walker, Heiko Strathmann, Viktor Gal, 
 *          Bjoern Esser, Sergey Lisitsyn, Roman Votyakov, Pan Deng
 */

#ifdef USE_GPL_SHOGUN

#include <shogun/lib/config.h>
#if defined(HAVE_NLOPT)
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
		    RegressionLabels* labels)
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
ModelSelectionParameters* build_tree(CInferenceMethod* inf,
				      LikelihoodModel* lik, CombinedKernel* kernel)
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c1 =
		new ModelSelectionParameters("inference_method", inf);

	root->append_child(c1);

        ModelSelectionParameters* c2 = new ModelSelectionParameters("scale");
        c1 ->append_child(c2);
        c2->build_values(0.99, 1.01, R_LINEAR);


	ModelSelectionParameters* c3 =
		new ModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3);

	ModelSelectionParameters* c4=new ModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(0.001, 1.0, R_LINEAR);

	ModelSelectionParameters* c5 =
		new ModelSelectionParameters("kernel", kernel);
	c1->append_child(c5);
	List* list = kernel->get_list();
	ModelSelectionParameters* cc1 = new ModelSelectionParameters("kernel_list", list);
	c5->append_child(cc1);

	ListElement* first = NULL;
	SGObject* k = list->get_first_element(first);

	ModelSelectionParameters* cc2 = new ModelSelectionParameters("first", first);
	cc1->append_child(cc2);

	Kernel* sub_kernel1 = kernel->get_kernel(0);
	ModelSelectionParameters* cc3 = new ModelSelectionParameters("data", sub_kernel1);
	cc2->append_child(cc3);

	ListElement* second = first;
	k = list->get_next_element(second);

	ModelSelectionParameters* cc4 = new ModelSelectionParameters("next", second);
	cc2->append_child(cc4);

	Kernel* sub_kernel2 = kernel->get_kernel(1);
	ModelSelectionParameters* cc5 = new ModelSelectionParameters("data", sub_kernel2);
	cc4->append_child(cc5);

	ListElement* third = second;
	k = list->get_next_element(third);

	ModelSelectionParameters* cc6 = new ModelSelectionParameters("next", third);
	cc4->append_child(cc6);

	Kernel* sub_kernel3 = kernel->get_kernel(2);
	ModelSelectionParameters* cc7 = new ModelSelectionParameters("data", sub_kernel3);
	cc6->append_child(cc7);

	ModelSelectionParameters* c6 =
		new ModelSelectionParameters("width");

	cc3->append_child(c6);
	c6->build_values(1.0, 4.0, R_LINEAR);

	ModelSelectionParameters* c66 =
		new ModelSelectionParameters("combined_kernel_weight");

	cc3->append_child(c66);
	c66->build_values(0.001, 1.0, R_LINEAR);

	ModelSelectionParameters* c7 =
		new ModelSelectionParameters("width");

	cc5->append_child(c7);
	c7->build_values(1.0, 4.0, R_LINEAR);

	ModelSelectionParameters* c77 =
		new ModelSelectionParameters("combined_kernel_weight");

	cc5->append_child(c77);
	c77->build_values(0.001, 1.0, R_LINEAR);

	ModelSelectionParameters* c8 =
		new ModelSelectionParameters("width");
	cc7->append_child(c8);
	c8->build_values(1.0, 4.0, R_LINEAR);

	ModelSelectionParameters* c88 =
		new ModelSelectionParameters("combined_kernel_weight");
	cc7->append_child(c88);
	c88->build_values(0.001, 1.0, R_LINEAR);


	return root;
}
*/

int main(int argc, char **argv)
{
	/* create some data and labels */
	SGMatrix<float64_t> matrix =
		 SGMatrix<float64_t>(dim_vectors, num_vectors);

	SGMatrix<float64_t> matrix2 =
		SGMatrix<float64_t>(dim_vectors, num_vectors);

	RegressionLabels* labels=new RegressionLabels(num_vectors);

	build_matrices(matrix2, matrix, labels);

	/* create training features */
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t> ();
	features->set_feature_matrix(matrix);

	CombinedFeatures* comb_features=new CombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);


	CombinedKernel* test_kernel = new CombinedKernel();
	GaussianKernel* sub_kernel1 = new GaussianKernel(10, 2);
	GaussianKernel* sub_kernel2 = new GaussianKernel(10, 2);
	GaussianKernel* sub_kernel3 = new GaussianKernel(10, 2);

	test_kernel->append_kernel(sub_kernel1);
	test_kernel->append_kernel(sub_kernel2);
	test_kernel->append_kernel(sub_kernel3);


	/*Allocate our Mean Function*/
	ZeroMean* mean = new ZeroMean();

	/*Allocate our Likelihood Model*/
	GaussianLikelihood* lik = new GaussianLikelihood();

	/*Allocate our inference method*/
	ExactInferenceMethod* inf =
			new ExactInferenceMethod(test_kernel,
						  comb_features, mean, labels, lik);


	/*Finally use these to allocate the Gaussian Process Object*/
	GaussianProcessRegression* gp =
		new GaussianProcessRegression(inf);


	//ModelSelectionParameters* root = build_tree(inf, lik, test_kernel);
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

	//GradientResult* result=(GradientResult*)grad->evaluate();

	//if(result->get_result_type() != GRADIENTEVALUATION_RESULT)
	//	SG_SERROR("Evaluation result not a GradientEvaluationResult!");

	//result->print_result();

	//SGVector<float64_t> alpha = inf->get_alpha();
	//SGVector<float64_t> labe = labels->get_labels();
	//SGVector<float64_t> diagonal = inf->get_diagonal_vector();
	//SGMatrix<float64_t> cholesky = inf->get_cholesky();
	//gp->set_return_type(GaussianProcessRegression::GP_RETURN_COV);

	//RegressionLabels* covariance = gp->apply_regression(comb_features);

	//gp->set_return_type(GaussianProcessRegression::GP_RETURN_MEANS);
	//
	//RegressionLabels* predictions = gp->apply_regression();

	//alpha.display_vector("Alpha Vector");
	//labe.display_vector("Labels");
	//diagonal.display_vector("sW Matrix");
	//covariance->get_labels().display_vector("Predicted Variances");
	//predictions->get_labels().display_vector("Mean Predictions");
	//cholesky.display_matrix("Cholesky Matrix L");
	//matrix.display_matrix("Training Features");
	//matrix2.display_matrix("Testing Features");

	///*free memory*/

	return 0;

}
#else // HAVE_NLOPT
int main(int argc, char **argv)
{
	return 0;
}
#endif // HAVE_NLOPT

#else //USE_GPL_SHOGUN
int main(int argc, char **argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
