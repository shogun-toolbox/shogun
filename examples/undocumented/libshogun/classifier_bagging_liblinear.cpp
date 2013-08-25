/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 */

#include <shogun/base/init.h>
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
	init_shogun_with_defaults();

	float64_t difference = 2.5;
	index_t dim = 2;
	index_t num_neg = 20;
	index_t num_pos = 20;
	int32_t num_bags = 5;
	int32_t bag_size = 25;
	
	/* streaming data generator for mean shift distributions */
    CMeanShiftDataGenerator* gen_n = new CMeanShiftDataGenerator(0, dim);
	CMeanShiftDataGenerator* gen_p = new CMeanShiftDataGenerator(difference, dim);

	CFeatures* neg = gen_n->get_streamed_features(num_pos);
	CFeatures* pos = gen_p->get_streamed_features(num_neg);
	CDenseFeatures<float64_t>* train_feats = 
		CDenseFeatures<float64_t>::obtain_from_generic(neg->create_merged_copy(pos));

	SGVector<float64_t> tl(num_neg+num_pos);
	tl.set_const(1);
	for (index_t i = 0; i < num_neg; ++i)
		tl[i] = -1;
	CBinaryLabels* train_labels = new CBinaryLabels(tl);
	
	CBaggingMachine* bm = new CBaggingMachine(train_feats, train_labels);
	CLibLinear* ll = new CLibLinear();
	ll->set_bias_enabled(true);
	CMajorityVote* mv = new CMajorityVote();

	bm->set_num_bags(num_bags);
	bm->set_bag_size(bag_size);
	bm->set_machine(ll);
	bm->set_combination_rule(mv);

	bm->train();

	CBinaryLabels* pred_bagging = bm->apply_binary(train_feats);
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();
	pred_bagging->get_int_labels().display_vector();
	
	float64_t bag_accuracy = eval->evaluate(pred_bagging, train_labels);
	float64_t oob_error = bm->get_oob_error(eval);

	CLibLinear* libLin = new CLibLinear(2.0, train_feats, train_labels);
	libLin->set_bias_enabled(true);
	libLin->train();
	CBinaryLabels* pred_liblin = libLin->apply_binary(train_feats);
	pred_liblin->get_int_labels().display_vector();
	float64_t liblin_accuracy = eval->evaluate(pred_liblin, train_labels);

	SG_SPRINT("bagging accuracy: %f (OOB-error: %f)\nLibLinear accuracy: %f\n", 
		bag_accuracy, oob_error, liblin_accuracy);

	SG_UNREF(bm);
	SG_UNREF(pos);
	SG_UNREF(neg);
	SG_UNREF(eval);

	exit_shogun();

	return 0;
}
