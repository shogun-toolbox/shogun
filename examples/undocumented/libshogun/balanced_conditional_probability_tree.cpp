/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Fernando Iglesias, Thoralf Klein, Shashwat Lal Das
 */

#include <shogun/lib/common.h>

#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/multiclass/tree/BalancedConditionalProbabilityTree.h>

using namespace shogun;

int main(int argc, char **argv)
{
	const char* train_file_name = "../data/7class_example4_train.dense";
	const char* test_file_name = "../data/7class_example4_test.dense";
	auto train_file = std::make_shared<StreamingAsciiFile>(train_file_name);

	auto train_features = std::make_shared<StreamingDenseFeatures<float32_t>>(train_file, true, 1024);

	auto cpt = std::make_shared<BalancedConditionalProbabilityTree>();
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

	auto test_file = std::make_shared<StreamingAsciiFile>(test_file_name);
	auto test_features = std::make_shared<StreamingDenseFeatures<float32_t>>(test_file, true, 1024);

	auto pred = cpt->apply_multiclass(test_features);
	test_features->reset_stream();
	SG_SPRINT("num_labels = %d\n", pred->get_num_labels());

	test_file = std::make_shared<StreamingAsciiFile>(test_file_name);
	test_features = std::make_shared<StreamingDenseFeatures<float32_t>>(test_file, true, 1024);

	auto gnd = std::make_shared<MulticlassLabels>(pred->get_num_labels());
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

	return 0;
}
