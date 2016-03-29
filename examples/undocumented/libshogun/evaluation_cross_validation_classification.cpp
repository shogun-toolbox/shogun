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
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test_cross_validation()
{
	/* data matrix dimensions */
	index_t num_vectors=40;
	index_t num_features=5;

	/* data means -1, 1 in all components, std deviation of 3 */
	SGVector<float64_t> mean_1(num_features);
	SGVector<float64_t> mean_2(num_features);
	SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -1.0);
	SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 1.0);
	float64_t sigma=3;

	SGVector<float64_t>::display_vector(mean_1.vector, mean_1.vlen, "mean 1");
	SGVector<float64_t>::display_vector(mean_2.vector, mean_2.vlen, "mean 2");

	/* fill data matrix around mean */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<num_features; ++j)
		{
			float64_t mean=i<num_vectors/2 ? mean_1.vector[0] : mean_2.vector[0];
			train_dat.matrix[i*num_features+j]=CMath::normal_random(mean, sigma);
		}
	}

	/* training features */
	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(train_dat);
	SG_REF(features);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
		lab.vector[i]=i<num_vectors/2 ? -1.0 : 1.0;

	CBinaryLabels* labels=new CBinaryLabels(lab);

	/* gaussian kernel */
	int32_t kernel_cache=100;
	int32_t width=10;
	CGaussianKernel* kernel=new CGaussianKernel(kernel_cache, width);
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=10;
	float64_t svm_eps=0.0001;
	CLibSVM* svm=new CLibSVM(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	/* train and output */
	svm->train(features);
	CBinaryLabels* output=CLabelsFactory::to_binary(svm->apply(features));
	for (index_t i=0; i<num_vectors; ++i)
		SG_SPRINT("i=%d, class=%f,\n", i, output->get_label(i));

	/* evaluation criterion */
	CContingencyTableEvaluation* eval_crit=
			new CContingencyTableEvaluation(ACCURACY);

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
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	CCrossValidation* cross=new CCrossValidation(svm, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(10);
//	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CrossValidationResult!");

	result->print_result();

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	sg_io->set_loglevel(MSG_DEBUG);

	test_cross_validation();

	exit_shogun();

	return 0;
}

