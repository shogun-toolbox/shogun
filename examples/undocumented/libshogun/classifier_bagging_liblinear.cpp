/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Bjoern Esser
 */

#include <shogun/machine/BaggingMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>

using namespace shogun;

int main(int argc, char** argv)
{
	float64_t difference = 2.5;
	index_t dim = 2;
	index_t num_neg = 20;
	index_t num_pos = 20;
	int32_t num_bags = 5;
	int32_t bag_size = 25;

	/* streaming data generator for mean shift distributions */
    MeanShiftDataGenerator* gen_n = new MeanShiftDataGenerator(0, dim);
	MeanShiftDataGenerator* gen_p = new MeanShiftDataGenerator(difference, dim);

	Features* neg = gen_n->get_streamed_features(num_pos);
	Features* pos = gen_p->get_streamed_features(num_neg);
	DenseFeatures<float64_t>* train_feats =
		neg->create_merged_copy(pos)->as<DenseFeatures<float64_t>>();

	SGVector<float64_t> tl(num_neg+num_pos);
	tl.set_const(1);
	for (index_t i = 0; i < num_neg; ++i)
		tl[i] = -1;
	BinaryLabels* train_labels = new BinaryLabels(tl);

	BaggingMachine* bm = new BaggingMachine(train_feats, train_labels);
	LibLinear* ll = new LibLinear();
	ll->set_bias_enabled(true);
	MajorityVote* mv = new MajorityVote();

	bm->set_num_bags(num_bags);
	bm->set_bag_size(bag_size);
	bm->set_machine(ll);
	bm->set_combination_rule(mv);

	bm->train();

	BinaryLabels* pred_bagging = bm->apply_binary(train_feats);
	ContingencyTableEvaluation* eval = new ContingencyTableEvaluation();
	pred_bagging->get_int_labels().display_vector();

	float64_t bag_accuracy = eval->evaluate(pred_bagging, train_labels);
	float64_t oob_error = bm->get_oob_error(eval);

	LibLinear* libLin = new LibLinear(2.0, train_feats, train_labels);
	libLin->set_bias_enabled(true);
	libLin->train();
	BinaryLabels* pred_liblin = libLin->apply_binary(train_feats);
	pred_liblin->get_int_labels().display_vector();
	float64_t liblin_accuracy = eval->evaluate(pred_liblin, train_labels);

	SG_SPRINT("bagging accuracy: %f (OOB-error: %f)\nLibLinear accuracy: %f\n",
		bag_accuracy, oob_error, liblin_accuracy);


	return 0;
}
