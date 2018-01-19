/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein
 */

#include <shogun/base/init.h>
#include <shogun/lib/common.h>

#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/classifier/svm/OnlineLibLinear.h>

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
