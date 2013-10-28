/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 yoo, thereisnoknife@gmail.com
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

/* cross-validation instances */
const index_t n_folds=2;
const index_t n_runs=2;

/* file data */
const char fname_feats[]="../data/fm_train_real.dat";
const char fname_labels[]="../data/label_train_multiclass.dat";

void test_multiclass_mkl_cv()
{
	/* init random number generator for reproducible results of cross-validation in the light of ASSERT(result->mean>0.81); some lines down below */
	sg_rand->set_seed(12);

	/* dense features from matrix */
	CCSVFile* feature_file = new CCSVFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);
	SG_UNREF(feature_file);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	/* labels from vector */
	CCSVFile* label_file = new CCSVFile(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);
	SG_UNREF(label_file);

	CMulticlassLabels* labels=new CMulticlassLabels(label_vec);
	SG_REF(labels);

	/* combined features and kernel */
	CCombinedFeatures *cfeats=new CCombinedFeatures();
	CCombinedKernel *cker=new CCombinedKernel();
	SG_REF(cfeats);
	SG_REF(cker);

	/** 1st kernel: gaussian */
	cfeats->append_feature_obj(features);
	cker->append_kernel(new CGaussianKernel(features, features, 1.2, 10));

	/** 2nd kernel: linear */
	cfeats->append_feature_obj(features);
	cker->append_kernel(new CLinearKernel(features, features));

	/** 3rd kernel: poly */
	cfeats->append_feature_obj(features);
	cker->append_kernel(new CPolyKernel(features, features, 2, true, 10));

	cker->init(cfeats, cfeats);

	/* create mkl instance */
	CMKLMulticlass* mkl=new CMKLMulticlass(1.2, cker, labels);
	SG_REF(mkl);
	mkl->set_epsilon(0.00001);
	mkl->parallel->set_num_threads(1);
	mkl->set_mkl_epsilon(0.001);
	mkl->set_mkl_norm(1.5);

	/* train to see weights */
	mkl->train();
	cker->get_subkernel_weights().display_vector("weights");

	CMulticlassAccuracy* eval_crit=new CMulticlassAccuracy();
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);
	CCrossValidation *cross=new CCrossValidation(mkl, cfeats, labels, splitting,
			eval_crit);
	cross->set_autolock(false);
	cross->set_num_runs(n_runs);
	cross->set_conf_int_alpha(0.05);

	/* perform x-val and print result */
	CCrossValidationResult* result=(CCrossValidationResult*)cross->evaluate();
	SG_SPRINT("mean of %d %d-fold x-val runs: %f\n", n_runs, n_folds,
			result->mean);

	/* assert high accuracy */
	ASSERT(result->mean>0.81);

	/* clean up */
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(cfeats);
	SG_UNREF(cker);
	SG_UNREF(mkl);
	SG_UNREF(cross);
	SG_UNREF(result);
}

int main(int argc, char** argv){
	shogun::init_shogun_with_defaults();

	// sg_io->set_loglevel(MSG_DEBUG);

	/* performs cross-validation on a multi-class mkl machine */
	test_multiclass_mkl_cv();

	exit_shogun();
}

