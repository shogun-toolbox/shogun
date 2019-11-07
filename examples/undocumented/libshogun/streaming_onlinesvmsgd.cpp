/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Shashwat Lal Das
 */

#include <shogun/lib/common.h>

#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/features/StreamingSparseFeatures.h>
#include <shogun/classifier/svm/OnlineSVMSGD.h>

using namespace shogun;

int main()
{
	// Create a StreamingAsciiFile from the training data
	char* train_file_name = "../data/train_sparsereal.light";
	StreamingAsciiFile* train_file = new StreamingAsciiFile(train_file_name);

	// Create a StreamingSparseFeatures from the StreamingAsciiFile.
	// The bool value is true if examples are labelled.
	// 1024 is a good standard value for the number of examples for the parser to hold at a time.
	StreamingSparseFeatures<float64_t>* train_features = new StreamingSparseFeatures<float64_t>(train_file, true, 1024);

	// Create an OnlineSVMSGD object from the features. The first parameter is 'C'.
	COnlineSVMSGD* sgd = new COnlineSVMSGD(1, train_features);

	sgd->set_bias_enabled(false); // Enable/disable bias
	sgd->set_lambda(0.1);	// Choose lambda
	sgd->train();		// Train

	train_file->close();

	// Now we want to test on other data
	char* test_file_name = "../data/fm_test_sparsereal.dat";
	StreamingAsciiFile* test_file = new StreamingAsciiFile(test_file_name);

	// Similar, but 'false' since the file contains unlabelled examples
	StreamingSparseFeatures<float64_t>* test_features = new StreamingSparseFeatures<float64_t>(test_file, false, 1024);

	// Apply on all examples and return a Labels*
	Labels* test_labels = sgd->apply(test_features);

	for (int32_t i=0; i<test_labels->get_num_labels(); i++)
		SG_SPRINT("For example %d, predicted label is %f.\n", i, test_labels->get_label(i));


	return 0;
}
