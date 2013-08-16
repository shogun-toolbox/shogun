/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanSquaredError.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

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
		lab.vector[i]=i+CMath::normal_random(0, 1.0);

	}

	/* training features */
	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	/* training labels */
	CRegressionLabels* labels=new CRegressionLabels(lab);

	/* kernel */
	CLinearKernel* kernel=new CLinearKernel();
	kernel->init(features, features);

	/* kernel ridge regression*/
	float64_t tau=0.0001;
	CKernelRidgeRegression* krr=new CKernelRidgeRegression(tau, kernel, labels);

	/* evaluation criterion */
	CMeanSquaredError* eval_crit=
			new CMeanSquaredError();

	/* train and output */
	krr->train(features);
	CRegressionLabels* output= CLabelsFactory::to_regression(krr->apply());
	for (index_t i=0; i<num_vectors; ++i)
	{
		SG_SPRINT("x=%f, train=%f, predict=%f\n", train_dat.matrix[i],
				labels->get_label(i), output->get_label(i));
	}

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training error: %f\n", eval_result);
	SG_UNREF(output);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=5;
	CCrossValidationSplitting* splitting=
			new CCrossValidationSplitting(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	CCrossValidation* cross=new CCrossValidation(krr, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(100);
	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	SG_SPRINT("cross_validation estimate:\n");
	result->print_result();

	/* same crude assertion as for above evaluation */
	ASSERT(result->mean<2);

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
#endif /* HAVE_LAPACK */
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_cross_validation();

	exit_shogun();

	return 0;
}

