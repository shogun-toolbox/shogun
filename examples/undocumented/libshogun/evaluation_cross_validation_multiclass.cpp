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
#include <shogun/features/Labels.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/StreamingDenseFeatures.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test_cross_validation()
{
	int32_t num_vectors = 0;
	int32_t num_feats   = 2;
	
	// Prepare to read a file for the training data
	char fname_feats[]  = "../data/fm_train_real.dat";
	char fname_labels[] = "../data/label_train_multiclass.dat";
	CStreamingAsciiFile* ffeats_train  = new CStreamingAsciiFile(fname_feats);
	CStreamingAsciiFile* flabels_train = new CStreamingAsciiFile(fname_labels);
	SG_REF(ffeats_train);
	SG_REF(flabels_train);

	CStreamingDenseFeatures< float64_t >* stream_features = 
		new CStreamingDenseFeatures< float64_t >(ffeats_train, false, 1024);

	CStreamingDenseFeatures< float64_t >* stream_labels = 
		new CStreamingDenseFeatures< float64_t >(flabels_train, true, 1024);

	SG_REF(stream_features);
	SG_REF(stream_labels);

	// Create a matrix with enough space to read all the feature vectors
	SGMatrix< float64_t > mat = SGMatrix< float64_t >(num_feats, 1000);

	// Read the values from the file and store them in mat
	SGVector< float64_t > vec;
	stream_features->start_parser();
	while ( stream_features->get_next_example() )
	{
		vec = stream_features->get_vector();

		for ( int32_t i = 0 ; i < num_feats ; ++i )
			mat.matrix[num_vectors*num_feats + i] = vec[i];

		num_vectors++;
		stream_features->release_example();
	}
	stream_features->end_parser();
	mat.num_cols = num_vectors;
	
	// Create features with the useful values from mat
	CDenseFeatures< float64_t >* features = new CDenseFeatures<float64_t>(mat);

	CLabels* labels = new CLabels(num_vectors);
	SG_REF(features);
	SG_REF(labels);

	// Read the labels from the file
	int32_t idx = 0;
	stream_labels->start_parser();
	while ( stream_labels->get_next_example() )
	{
		labels->set_int_label( idx++, (int32_t)stream_labels->get_label() );
		stream_labels->release_example();
	}
	stream_labels->end_parser();

	/* create svm via libsvm */
	float64_t svm_C=10;
	float64_t svm_eps=0.0001;
	CMulticlassLibLinear* svm=new CMulticlassLibLinear(svm_C, features, labels);
	svm->set_epsilon(svm_eps);

	/* train and output */
	svm->train(features);
	CLabels* output=svm->apply(features);
	for (index_t i=0; i<num_vectors; ++i)
		SG_SPRINT("i=%d, class=%f,\n", i, output->get_label(i));

	/* evaluation criterion */
	CMulticlassAccuracy* eval_crit = new CMulticlassAccuracy ();

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);
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
	cross->set_conf_int_alpha(0.05);

	/* actual evaluation */
	CrossValidationResult result=cross->evaluate();
	result.print_result();

	/* clean up */
	SG_UNREF(cross);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(ffeats_train);
	SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	sg_io->set_loglevel(MSG_DEBUG);

	test_cross_validation();

	exit_shogun();

	return 0;
}

