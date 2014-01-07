/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */
#include <kernel/GaussianKernel.h>
#include <kernel/CombinedKernel.h>
#include <labels/BinaryLabels.h>
#include <features/DenseFeatures.h>
#include <classifier/mkl/MKLClassification.h>
#include <classifier/svm/LibSVM.h>
#include <evaluation/CrossValidation.h>
#include <evaluation/CrossValidationPrintOutput.h>
#include <evaluation/CrossValidationMKLStorage.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <evaluation/ContingencyTableEvaluation.h>
#include <mathematics/Statistics.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> lab, SGMatrix<float64_t> feat,
		float64_t dist)
{
	index_t dims=feat.num_rows;
	index_t num=lab.vlen;

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=CMath::random(0.0, 1.0)-dist;
		}
	}
	lab.display_vector("lab");
	feat.display_matrix("feat");
}

void test_mkl_cross_validation()
{
	/* generate random data */
	index_t num=10;
	index_t dims=2;
	float64_t dist=0.5;
	SGVector<float64_t> lab(num);
	SGMatrix<float64_t> feat(dims, num);
	gen_rand_data(lab, feat, dist);

	/*create train labels */
	CLabels* labels=new CBinaryLabels(lab);

	/* create train features */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	features->set_feature_matrix(feat);
	SG_REF(features);

	/* create combined features */
	CCombinedFeatures* comb_features=new CCombinedFeatures();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	SG_REF(comb_features);

	/* create multiple gaussian kernels */
	CCombinedKernel* kernel=new CCombinedKernel();
	kernel->append_kernel(new CGaussianKernel(10, 0.1));
	kernel->append_kernel(new CGaussianKernel(10, 1));
	kernel->append_kernel(new CGaussianKernel(10, 2));
	kernel->init(comb_features, comb_features);
	SG_REF(kernel);

	/* create mkl using libsvm, due to a mem-bug, interleaved is not possible */
	CMKLClassification* svm=new CMKLClassification(new CLibSVM());
	svm->set_interleaved_optimization_enabled(false);
	svm->set_kernel(kernel);
	SG_REF(svm);

	/* create cross-validation instance */
	index_t num_folds=3;
	CSplittingStrategy* split=new CStratifiedCrossValidationSplitting(labels,
			num_folds);
	CEvaluation* eval=new CContingencyTableEvaluation(ACCURACY);
	CCrossValidation* cross=new CCrossValidation(svm, comb_features, labels, split, eval, false);

	/* add print output listener and mkl storage listener */
	cross->add_cross_validation_output(new CCrossValidationPrintOutput());
	CCrossValidationMKLStorage* mkl_storage=new CCrossValidationMKLStorage();
	cross->add_cross_validation_output(mkl_storage);

	/* perform cross-validation, this will print loads of information
	 * (caused by the CCrossValidationPrintOutput instance attached to it) */
	CEvaluationResult* result=cross->evaluate();

	/* print mkl weights */
	SGMatrix<float64_t> weights=mkl_storage->get_mkl_weights();
	weights.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	CStatistics::matrix_mean(weights, false).display_vector("mean per kernel");
	CStatistics::matrix_variance(weights, false).display_vector("variance per kernel");
	CStatistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	SG_UNREF(result);

	/* again for two runs */
	cross->set_num_runs(2);
	result=cross->evaluate();

	/* print mkl weights */
	weights=mkl_storage->get_mkl_weights();
	weights.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	CStatistics::matrix_mean(weights, false).display_vector("mean per kernel");
	CStatistics::matrix_variance(weights, false).display_vector("variance per kernel");
	CStatistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	/* clean up */
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(comb_features);
	SG_UNREF(svm);
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test_mkl_cross_validation();

	exit_shogun();
	return 0;
}

