/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Jacob Walker, Giovanni De Toni, 
 *          Evgeniy Andreev, Soumyajit De, Viktor Gal, Sergey Lisitsyn
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanSquaredError.h>

using namespace shogun;

void test_cross_validation()
{
#ifdef HAVE_LAPACK
	/* data matrix dimensions */
	index_t num_vectors=100;
	index_t num_features=1;

	/* training label data */
	SGVector<float64_t> lab(num_vectors);

	/* fill data matrix and labels */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	SGVector<float64_t>::range_fill_vector(train_dat.matrix, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		/* labels are linear plus noise */
		lab.vector[i]=i+Math::normal_random(0, 1.0);

	}

	/* training features */
	auto features=std::make_shared<DenseFeatures<float64_t>>(train_dat);

	/* training labels */
	auto labels=std::make_shared<RegressionLabels>(lab);

	/* kernel */
	auto kernel=std::make_shared<LinearKernel>();
	kernel->init(features, features);

	/* kernel ridge regression*/
	float64_t tau=0.0001;
	auto krr=std::make_shared<KernelRidgeRegression>(tau, kernel, labels);

	/* evaluation criterion */
	auto eval_crit=std::make_shared<MeanSquaredError>();

	/* train and output */
	krr->train(features);
	auto output = krr->apply()->as<RegressionLabels>();
	for (index_t i=0; i<num_vectors; ++i)
	{
		SG_SPRINT("x=%f, train=%f, predict=%f\n", train_dat.matrix[i],
				labels->get_label(i), output->get_label(i));
	}

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training error: %f\n", eval_result);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=5;
	auto splitting=
		std::make_shared<CrossValidationSplitting>(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	auto cross=std::make_shared<CrossValidation>(krr, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(100);
//	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	auto result=cross->evaluate()->as<CrossValidationResult>();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	SG_SPRINT("cross_validation estimate:\n");
	result->print_result();

	/* same crude assertion as for above evaluation */
	ASSERT(result->get_mean() < 2);

#endif /* HAVE_LAPACK */
}

int main(int argc, char **argv)
{
	test_cross_validation();

	return 0;
}

