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

#include <io/streaming/StreamingAsciiFile.h>
#include <features/streaming/StreamingDenseFeatures.h>
#include <classifier/svm/OnlineLibLinear.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	// Create a StreamingAsciiFile from the training data
	const char* train_file_name = "../data/train_densereal.light";
	CStreamingAsciiFile* train_file = new CStreamingAsciiFile(train_file_name);
	SG_REF(train_file);

	// The bool value is true if examples are labelled.
	// 1024 is a good standard value for the number of examples for the parser to hold at a time.
	CStreamingDenseFeatures<float32_t>* train_features = new CStreamingDenseFeatures<float32_t>(train_file, true, 1024);
	SG_REF(train_features);

	// Create an OnlineLiblinear object from the features. The first parameter is 'C'.
	COnlineLibLinear* svm = new COnlineLibLinear(1, train_features);

	svm->set_bias_enabled(false); // Enable/disable bias
	svm->train();		// Train

	train_file->close();

	// Now we want to test on other data
	const char* test_file_name = "../data/fm_test_densereal.dat";
	CStreamingAsciiFile* test_file = new CStreamingAsciiFile(test_file_name);
	SG_REF(test_file);

	// Similar, but 'false' since the file contains unlabelled examples
	CStreamingDenseFeatures<float64_t>* test_features = new CStreamingDenseFeatures<float64_t>(test_file, false, 1024);
	SG_REF(test_features);

	// Apply on all examples and return a CLabels*
	CRegressionLabels* test_labels = svm->apply_regression(test_features);

	for (int32_t i=0; i<test_labels->get_num_labels(); i++)
		SG_SPRINT("For example %d, predicted label is %f.\n", i, test_labels->get_label(i));

	SG_UNREF(test_features);
	SG_UNREF(test_labels);
	SG_UNREF(test_file);
	SG_UNREF(train_features);
	SG_UNREF(train_file);
	SG_UNREF(svm);

	exit_shogun();

	return 0;
}
