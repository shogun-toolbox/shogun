/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * This example demonstrates use of the Vowpal Wabbit learning algorithm.
 */

#include <lib/common.h>

#include <io/streaming/StreamingAsciiFile.h>
#include <features/streaming/StreamingDenseFeatures.h>
#include <multiclass/tree/BalancedConditionalProbabilityTree.h>

using namespace shogun;

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	const char* train_file_name = "../data/7class_example4_train.dense";
	const char* test_file_name = "../data/7class_example4_test.dense";
	CStreamingAsciiFile* train_file = new CStreamingAsciiFile(train_file_name);
	SG_REF(train_file);

	CStreamingDenseFeatures<float32_t>* train_features = new CStreamingDenseFeatures<float32_t>(train_file, true, 1024);
	SG_REF(train_features);

	CBalancedConditionalProbabilityTree *cpt = new CBalancedConditionalProbabilityTree();
	cpt->set_num_passes(1);
	cpt->set_features(train_features);

	if (argc > 1)
	{
		float64_t alpha = 0.5;
		sscanf(argv[1], "%lf", &alpha);
		SG_SPRINT("Setting alpha to %.2lf\n", alpha);
		cpt->set_alpha(alpha);
	}

	cpt->train();
	cpt->print_tree();

	CStreamingAsciiFile* test_file = new CStreamingAsciiFile(test_file_name);
	SG_REF(test_file);
	CStreamingDenseFeatures<float32_t>* test_features = new CStreamingDenseFeatures<float32_t>(test_file, true, 1024);
	SG_REF(test_features);

	CMulticlassLabels *pred = cpt->apply_multiclass(test_features);
	test_features->reset_stream();
	SG_SPRINT("num_labels = %d\n", pred->get_num_labels());

	SG_UNREF(test_features);
	SG_UNREF(test_file);
	test_file = new CStreamingAsciiFile(test_file_name);
	SG_REF(test_file);
	test_features = new CStreamingDenseFeatures<float32_t>(test_file, true, 1024);
	SG_REF(test_features);

	CMulticlassLabels *gnd = new CMulticlassLabels(pred->get_num_labels());
	SG_REF(gnd);
	test_features->start_parser();
	for (int32_t i=0; i < pred->get_num_labels(); ++i)
	{
		test_features->get_next_example();
		gnd->set_int_label(i, test_features->get_label());
		test_features->release_example();
	}
	test_features->end_parser();

	int32_t n_correct = 0;
	for (index_t i=0; i < pred->get_num_labels(); ++i)
	{
		if (pred->get_int_label(i) == gnd->get_int_label(i))
			n_correct++;
		//SG_SPRINT("%d-%d ", pred->get_int_label(i), gnd->get_int_label(i));
	}
	SG_SPRINT("\n");

	SG_SPRINT("Multiclass Accuracy = %.2f%%\n", 100.0*n_correct / gnd->get_num_labels());

	SG_UNREF(gnd);
	SG_UNREF(train_features);
	SG_UNREF(test_features);
	SG_UNREF(train_file);
	SG_UNREF(test_file);
	SG_UNREF(cpt);
	SG_UNREF(pred);

	exit_shogun();

	return 0;
}
