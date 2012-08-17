/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 yoo, thereisnoknife@gmail.com
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

void test_multiclass_mkl_cv()
{
	/* stream data from a file */
	int32_t num_vectors=50;
	int32_t num_feats=2;

	/* file data */
	char fname_feats[]="../data/fm_train_real.dat";
	char fname_labels[]="../data/label_train_multiclass.dat";
	CStreamingAsciiFile* ffeats_train=new CStreamingAsciiFile(fname_feats);
	CStreamingAsciiFile* flabels_train=new CStreamingAsciiFile(fname_labels);
	SG_REF(ffeats_train);
	SG_REF(flabels_train);

	/* streaming data */
	CStreamingDenseFeatures<float64_t>* stream_features=
			new CStreamingDenseFeatures<float64_t>(ffeats_train, false, 1024);
	CStreamingDenseFeatures<float64_t>* stream_labels=
			new CStreamingDenseFeatures<float64_t>(flabels_train, true, 1024);
	SG_REF(stream_features);
	SG_REF(stream_labels);

	/* matrix data */
	SGMatrix<float64_t> mat=SGMatrix<float64_t>(num_feats, num_vectors);
	SGVector<float64_t> vec;
	stream_features->start_parser();

	index_t count=0;
	while (stream_features->get_next_example() && count<num_vectors)
	{
		vec=stream_features->get_vector();
		for (int32_t i=0; i<num_feats; ++i)
			mat(i,count)=vec[i];

		stream_features->release_example();
		count++;
	}
	stream_features->end_parser();
	mat.num_cols=num_vectors;

	/* dense features from streamed matrix */
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(mat);
	CMulticlassLabels* labels=new CMulticlassLabels(num_vectors);
	SG_REF(features);
	SG_REF(labels);

	/* read labels from file */
	int32_t idx=0;
	stream_labels->start_parser();
	while (stream_labels->get_next_example())
	{
		labels->set_int_label(idx++, (int32_t)stream_labels->get_label());
		stream_labels->release_example();
	}
	stream_labels->end_parser();

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

	/* cross-validation instances */
	index_t n_folds=3;
	index_t n_runs=5;
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
	ASSERT(result->mean>0.9);

	/* clean up */
	SG_UNREF(ffeats_train);
	SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
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

//	sg_io->set_loglevel(MSG_DEBUG);

	/* performs cross-validation on a multi-class mkl machine */
	test_multiclass_mkl_cv();

	exit_shogun();
}

