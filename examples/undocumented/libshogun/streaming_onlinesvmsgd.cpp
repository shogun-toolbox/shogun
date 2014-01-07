/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * This example demonstrates use of the online variant of SGD which
 * relies on the streaming features framework.
 */

#include <lib/common.h>

#include <io/StreamingAsciiFile.h>
#include <features/StreamingSparseFeatures.h>
#include <classifier/svm/OnlineSVMSGD.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	// Create a StreamingAsciiFile from the training data
	char* train_file_name = "../data/train_sparsereal.light";
	CStreamingAsciiFile* train_file = new CStreamingAsciiFile(train_file_name);
	SG_REF(train_file);

	// Create a StreamingSparseFeatures from the StreamingAsciiFile.
	// The bool value is true if examples are labelled.
	// 1024 is a good standard value for the number of examples for the parser to hold at a time.
	CStreamingSparseFeatures<float64_t>* train_features = new CStreamingSparseFeatures<float64_t>(train_file, true, 1024);
	SG_REF(train_features);

	// Create an OnlineSVMSGD object from the features. The first parameter is 'C'.
	COnlineSVMSGD* sgd = new COnlineSVMSGD(1, train_features);

	sgd->set_bias_enabled(false); // Enable/disable bias
	sgd->set_lambda(0.1);	// Choose lambda
	sgd->train();		// Train

	train_file->close();

	// Now we want to test on other data
	char* test_file_name = "../data/fm_test_sparsereal.dat";
	CStreamingAsciiFile* test_file = new CStreamingAsciiFile(test_file_name);
	SG_REF(test_file);

	// Similar, but 'false' since the file contains unlabelled examples
	CStreamingSparseFeatures<float64_t>* test_features = new CStreamingSparseFeatures<float64_t>(test_file, false, 1024);
	SG_REF(test_features);

	// Apply on all examples and return a CLabels*
	CLabels* test_labels = sgd->apply(test_features);

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
