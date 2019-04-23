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
	auto train_file = std::make_shared<StreamingAsciiFile>(train_file_name);

	// The bool value is true if examples are labelled.
	// 1024 is a good standard value for the number of examples for the parser to hold at a time.
	auto train_features = std::make_shared<StreamingDenseFeatures<float32_t>>(train_file, true, 1024);

	// Create an OnlineLiblinear object from the features. The first parameter is 'C'.
	auto svm = std::make_shared<OnlineLibLinear>(1, train_features);

	svm->set_bias_enabled(false); // Enable/disable bias
	svm->train();		// Train

	train_file->close();

	// Now we want to test on other data
	const char* test_file_name = "../data/fm_test_densereal.dat";
	auto test_file = std::make_shared<StreamingAsciiFile>(test_file_name);

	// Similar, but 'false' since the file contains unlabelled examples
	auto test_features = std::make_shared<StreamingDenseFeatures<float64_t>>(test_file, false, 1024);

	// Apply on all examples and return a Labels*
	auto test_labels = svm->apply_regression(test_features);

	for (int32_t i=0; i<test_labels->get_num_labels(); i++)
		SG_SPRINT("For example %d, predicted label is %f.\n", i, test_labels->get_label(i));


	exit_shogun();

	return 0;
}
