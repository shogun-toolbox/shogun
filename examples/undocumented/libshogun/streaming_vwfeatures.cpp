/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * This example demonstrates use of online SGD with CStreamingVwFeatures
 * as the features object.
 */

#include <lib/common.h>

#include <io/streaming/StreamingVwFile.h>
#include <features/streaming/StreamingVwFeatures.h>
#include <classifier/svm/OnlineSVMSGD.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	const char* train_file_name = "../data/train_sparsereal.light";
	CStreamingVwFile* train_file = new CStreamingVwFile(train_file_name);
	train_file->set_parser_type(T_SVMLIGHT); // Treat the file as SVMLight format
	SG_REF(train_file);

	CStreamingVwFeatures* train_features = new CStreamingVwFeatures(train_file, true, 1024);
	SG_REF(train_features);

	COnlineSVMSGD* sgd = new COnlineSVMSGD(1, train_features);
	sgd->set_bias_enabled(false);
	sgd->set_lambda(0.1);
	sgd->train();

	train_file->close();

	// Now we want to test on other data
	const char* test_file_name = "../data/fm_test_sparsereal.dat";
	CStreamingVwFile* test_file = new CStreamingVwFile(test_file_name);
	test_file->set_parser_type(T_SVMLIGHT);
	SG_REF(test_file);

	// Similar, but 'false' since the file contains unlabelled examples
	CStreamingVwFeatures* test_features = new CStreamingVwFeatures(test_file, false, 1024);
	SG_REF(test_features);

	// Apply on all examples and return a CLabels*
	CBinaryLabels* test_labels = sgd->apply_binary(test_features);

	for (int32_t i=0; i<test_labels->get_num_labels(); i++)
		SG_SPRINT("For example %d, predicted label is %f.\n", i, test_labels->get_label(i));

	SG_UNREF(test_features);
	SG_UNREF(test_file);
	SG_UNREF(train_features);
	SG_UNREF(train_file);
	SG_UNREF(sgd);

	exit_shogun();

	return 0;
}
