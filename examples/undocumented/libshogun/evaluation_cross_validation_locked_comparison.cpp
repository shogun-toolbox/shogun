/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <base/init.h>
#include <features/DenseFeatures.h>
#include <labels/BinaryLabels.h>
#include <kernel/GaussianKernel.h>
#include <classifier/svm/LibSVM.h>
#include <classifier/svm/SVMLight.h>
#include <evaluation/CrossValidation.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <evaluation/ContingencyTableEvaluation.h>
#include <lib/Time.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test_cross_validation()
{
	/* data matrix dimensions */
	index_t num_vectors=50;
	index_t num_features=5;

	/* data means -1, 1 in all components, std deviation of sigma */
	SGVector<float64_t> mean_1(num_features);
	SGVector<float64_t> mean_2(num_features);
	SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -1.0);
	SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 1.0);
	float64_t sigma=1.5;

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
	CGaussianKernel* kernel=new CGaussianKernel();
	kernel->set_width(10);
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=1;
	float64_t svm_eps=0.0001;
	CSVM* svm=new CLibSVM(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	/* train and output the normal way */
	SG_SPRINT("starting normal training\n");
	svm->train(features);
	CBinaryLabels* output=CLabelsFactory::to_binary(svm->apply(features));

	/* evaluation criterion */
	CContingencyTableEvaluation* eval_crit=
			new CContingencyTableEvaluation(ACCURACY);

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);
	SG_UNREF(output);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=3;
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	CCrossValidation* cross=new CCrossValidation(svm, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(5);
	cross->set_conf_int_alpha(0.05);

	CCrossValidationResult* tmp;
	/* no locking */
	index_t repetitions=5;
	SG_SPRINT("unlocked x-val\n");
	kernel->init(features, features);
	cross->set_autolock(false);
	CTime time;
	time.start();
	for (index_t i=0; i<repetitions; ++i)
	{
		tmp = (CCrossValidationResult*)cross->evaluate();
		SG_UNREF(tmp);
	}

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* auto_locking in every iteration of this loop (better, not so nice) */
	SG_SPRINT("locked in every iteration x-val\n");
	cross->set_autolock(true);
	time.start();

	for (index_t i=0; i<repetitions; ++i)
        {
                tmp = (CCrossValidationResult*)cross->evaluate();
                SG_UNREF(tmp);
        }

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* lock once before, (no locking/unlocking in this loop) */
	svm->data_lock(labels, features);
	SG_SPRINT("locked x-val\n");
	time.start();

        for (index_t i=0; i<repetitions; ++i)
        {
                tmp = (CCrossValidationResult*)cross->evaluate();
                SG_UNREF(tmp);
        }

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* clean up */
	SG_UNREF(cross);
	SG_UNREF(features);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_cross_validation();

	exit_shogun();

	return 0;
}

