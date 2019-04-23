/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang
 */

#include <shogun/lib/common.h>

#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/features/StreamingDenseFeatures.h>
#include <shogun/multiclass/tree/RandomConditionalProbabilityTree.h>

using namespace shogun;

int main()
{
	init_shogun_with_defaults();

	const char* train_file_name = "../data/7class_example4_train.dense";
	const char* test_file_name = "../data/7class_example4_test.dense";
	StreamingAsciiFile* train_file = new StreamingAsciiFile(train_file_name);

	StreamingDenseFeatures<float32_t>* train_features = new StreamingDenseFeatures<float32_t>(train_file, true, 1024);

	CRandomConditionalProbabilityTree *cpt = new CRandomConditionalProbabilityTree();
	cpt->set_num_passes(1);
	cpt->set_features(train_features);
	cpt->train();
	cpt->print_tree();

	StreamingAsciiFile* test_file = new StreamingAsciiFile(test_file_name);
	StreamingDenseFeatures<float32_t>* test_features = new StreamingDenseFeatures<float32_t>(test_file, true, 1024);

	MulticlassLabels *pred = cpt->apply_multiclass(test_features);
	test_features->reset_stream();
	SG_SPRINT("num_labels = %d\n", pred->get_num_labels());

	test_file = new StreamingAsciiFile(test_file_name);
	test_features = new StreamingDenseFeatures<float32_t>(test_file, true, 1024);

	MulticlassLabels *gnd = new MulticlassLabels(pred->get_num_labels());
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


	exit_shogun();

	return 0;
}
