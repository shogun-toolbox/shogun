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

#include <shogun/lib/common.h>

#include <shogun/io/StreamingVwFile.h>
#include <shogun/features/StreamingVwFeatures.h>
#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	const char* train_file_name = "../data/7class_example4_train.light";
	const char* test_file_name = "../data/7class_example4_test.light";
	CStreamingVwFile* train_file = new CStreamingVwFile(train_file_name);
	train_file->set_parser_type(T_SVMLIGHT); // Treat the file as SVMLight format
	SG_REF(train_file);

	CStreamingVwFeatures* train_features = new CStreamingVwFeatures(train_file, true, 1024);
	SG_REF(train_features);

	CRandomConditionalProbabilityTree *cpt = new CRandomConditionalProbabilityTree();
	cpt->set_num_passes(1);
	cpt->set_features(train_features);
	cpt->train();

	CStreamingVwFile* test_file = new CStreamingVwFile(test_file_name);
	test_file->set_parser_type(T_SVMLIGHT); // Treat the file as SVMLight format
	SG_REF(test_file);
	CStreamingVwFeatures* test_features = new CStreamingVwFeatures(test_file, true, 1024);
	SG_REF(test_features);

	CMulticlassLabels *pred = cpt->apply_multiclass(test_features);

	for (index_t i=0; i < pred->get_num_labels(); ++i)
		printf("%d ", pred->get_int_label(i));
	printf("\n");

	SG_UNREF(train_features);
	SG_UNREF(test_features);
	SG_UNREF(train_file);
	SG_UNREF(test_file);
	SG_UNREF(cpt);
	SG_UNREF(pred);

	exit_shogun();

	return 0;
}
