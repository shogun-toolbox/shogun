/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Roman Votyakov, Jacob Walker, Viktor Gal, 
 *          Bjoern Esser, Pan Deng
 */

#include <shogun/lib/config.h>

// temporally disabled, since API was changed
#if defined(HAVE_NLOPT) && 0

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/LaplacianInferenceMethod.h>
#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/evaluation/GradientCriterion.h>

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

ModelSelectionParameters* build_tree(CInferenceMethod* inf,
				      LikelihoodModel* lik, Kernel* kernel)
{
	ModelSelectionParameters* root=new ModelSelectionParameters();

	ModelSelectionParameters* c1 =
			new ModelSelectionParameters("inference_method", inf);
	root->append_child(c1);

	ModelSelectionParameters* c2 = new ModelSelectionParameters("scale");
	c1 ->append_child(c2);
	c2->build_values(0.5, 4.0, R_LINEAR);


	ModelSelectionParameters* c3 =
			new ModelSelectionParameters("likelihood_model", lik);
	c1->append_child(c3);

	ModelSelectionParameters* c4=new ModelSelectionParameters("sigma");
	c3->append_child(c4);
	c4->build_values(0.01, 4.0, R_LINEAR);

	ModelSelectionParameters* c43=new ModelSelectionParameters("df");
	c3->append_child(c43);
	c43->build_values(500.0, 1000.0, R_LINEAR);



	ModelSelectionParameters* c5 =
			new ModelSelectionParameters("kernel", kernel);
	c1->append_child(c5);

	ModelSelectionParameters* c6 =
			new ModelSelectionParameters("width");
	c5->append_child(c6);
	c6->build_values(0.01, 4.0, R_LINEAR);

	return root;
}

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

	/* create testing features */
	DenseFeatures<float64_t>* features2=new DenseFeatures<float64_t> ();
	features2->set_feature_matrix(matrix2);



	/*Allocate our Kernel*/
	GaussianKernel* test_kernel = new GaussianKernel(10, 2);

	test_kernel->init(features, features);

	/*Allocate our mean function*/
	ZeroMean* mean = new ZeroMean();

	/*Allocate our likelihood function*/
	StudentsTLikelihood* lik = new StudentsTLikelihood();

	/*Allocate our inference method*/
	CLaplacianInferenceMethod* inf =
			new CLaplacianInferenceMethod(test_kernel,
						  features, mean, labels, lik);


	/*Finally use these to allocate the Gaussian Process Object*/
	GaussianProcessRegression* gp =
			new GaussianProcessRegression(inf);


	/*Build the parameter tree for model selection*/
	ModelSelectionParameters* root = build_tree(inf, lik, test_kernel);

	/*Criterion for gradient search*/
	CGradientCriterion* crit = new CGradientCriterion();

	/*This will evaluate our inference method for its derivatives*/
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

	/*Search for best parameters*/
	CParameterCombination* best_combination=grad_search->select_model(true);

	/*Output all the results and information*/
	if (best_combination)
	{
		SG_SPRINT("best parameter(s):\n");
		best_combination->print_tree();

		best_combination->apply_to_machine(gp);
	}

	GradientResult* result=(GradientResult*)grad->evaluate();

	if(result->get_result_type() != GRADIENTEVALUATION_RESULT)
		SG_SERROR("Evaluation result not a GradientEvaluationResult!");

	result->print_result();

	SGVector<float64_t> alpha = inf->get_alpha();
	SGVector<float64_t> labe = labels->get_labels();
	SGVector<float64_t> diagonal = inf->get_diagonal_vector();
	SGMatrix<float64_t> cholesky = inf->get_cholesky();
	RegressionLabels* predictions=gp->apply_regression(features);
	SGVector<float64_t> variance_vector=gp->get_variance_vector(features);

	alpha.display_vector("Alpha Vector");
	labe.display_vector("Labels");
	diagonal.display_vector("sW Matrix");
	variance_vector.display_vector("Predicted Variances");
	predictions->get_labels().display_vector("Mean Predictions");
	cholesky.display_matrix("Cholesky Matrix L");
	matrix.display_matrix("Training Features");
	matrix2.display_matrix("Testing Features");

	/*free memory*/

	return 0;
}
#else
int main(int argc, char **argv)
{
	return 0;
}
#endif
