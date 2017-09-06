/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/RandomForest.h>
#include <stdio.h>

using namespace shogun;

class RandomForest : public ::testing::Test
{
public:
	CDenseFeatures<float64_t>* weather_features_test;
	CDenseFeatures<float64_t>* weather_features_train;
	CMulticlassLabels* weather_labels_train;
	SGVector<bool> weather_ft;

	virtual void SetUp()
	{
		sg_rand->set_seed(1);
		load_toy_data();
	}

	virtual void TearDown()
	{
		SG_UNREF(weather_features_train);
		SG_UNREF(weather_features_test);
		SG_UNREF(weather_labels_train);
	}

	void load_toy_data()
	{
		SGMatrix<float64_t> weather_data(4, 14);
		SGVector<float64_t> lab(14);

		generate_toy_data_weather(weather_data, lab);

		weather_features_train = new CDenseFeatures<float64_t>(weather_data);
		weather_labels_train = new CMulticlassLabels(lab);

		SGMatrix<float64_t> test(4, 5);
		SGVector<float64_t> test_labels(4);
		generate_toy_data_weather(test, test_labels, false);
		weather_features_test = new CDenseFeatures<float64_t>(test);

		auto feature_types = SGVector<bool>(4);

		feature_types[0] = true;
		feature_types[1] = true;
		feature_types[2] = true;
		feature_types[3] = true;

		weather_ft = feature_types;

		SG_REF(weather_features_train);
		SG_REF(weather_features_test);
		SG_REF(weather_labels_train);
	}
};

TEST_F(RandomForest, classify_nominal_test)
{
	CRandomForest* c =
	    new CRandomForest(weather_features_train, weather_labels_train, 100, 2);
	c->set_feature_types(weather_ft);
	CMajorityVote* mv = new CMajorityVote();
	c->set_combination_rule(mv);
	c->parallel->set_num_threads(1);
	c->train(weather_features_train);

	CMulticlassLabels* result =
	    (CMulticlassLabels*)c->apply(weather_features_test);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(0.0, res_vector[4]);

	CMulticlassAccuracy* eval=new CMulticlassAccuracy();
	EXPECT_NEAR(0.571429, c->get_oob_error(eval), 1e-6);

	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(eval);
}

TEST_F(RandomForest, classify_non_nominal_test)
{
	weather_ft[0] = false;
	weather_ft[1] = false;
	weather_ft[2] = false;
	weather_ft[3] = false;

	CRandomForest* c =
	    new CRandomForest(weather_features_train, weather_labels_train, 100, 2);
	c->set_feature_types(weather_ft);
	CMajorityVote* mv = new CMajorityVote();
	c->set_combination_rule(mv);
	c->parallel->set_num_threads(1);
	c->train(weather_features_train);

	CMulticlassLabels* result =
	    (CMulticlassLabels*)c->apply(weather_features_test);
	SGVector<float64_t> res_vector=result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);

	CMulticlassAccuracy* eval=new CMulticlassAccuracy();
	EXPECT_NEAR(0.642857, c->get_oob_error(eval), 1e-6);

	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(eval);
}

TEST_F(RandomForest, score_compare_sklearn_toydata)
{
	sg_rand->set_seed(1);
	// Comparison with sklearn's RandomForest probability outputs
	// https://github.com/scikit-learn/scikit-learn/blob/6f70202ef9beefd3db9bb028755a0c38b4c5c8e7/sklearn/ensemble/tests/test_voting_classifier.py#L143
	float64_t data_A[] = {-1.1, -1.5, -1.2, -1.4, -3.4, -2.2, 1.1, 1.2};
	float64_t expected_probabilities[] = {0.2, 0.2, 0.8, 0.7};

	SGMatrix<float64_t> data(data_A, 2, 4, false);

	CDenseFeatures<float64_t>* features_train =
	    new CDenseFeatures<float64_t>(data);

	float64_t labels[] = {0.0, 0.0, 1.0, 1.0};
	SGVector<float64_t> lab(labels, 4);
	CMulticlassLabels* labels_train = new CMulticlassLabels(lab);

	CRandomForest* c = new CRandomForest(features_train, labels_train, 10, 2);
	SGVector<bool> ft = SGVector<bool>(2);
	ft[0] = false;
	ft[1] = false;
	c->set_feature_types(ft);

	CMeanRule* mr = new CMeanRule();
	c->set_combination_rule(mr);
	c->train(features_train);

	auto result = c->apply_binary(features_train);
	SGVector<float64_t> res_vector = result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(-1.0, res_vector[0]);
	EXPECT_EQ(-1.0, res_vector[1]);
	EXPECT_EQ(+1.0, res_vector[2]);
	EXPECT_EQ(+1.0, res_vector[3]);

	for (auto i = 0; i < 4; ++i)
	{
		EXPECT_NEAR(expected_probabilities[i], values_vector[i], 1.1e-1);
	}

	SG_UNREF(result);
}

TEST_F(RandomForest, score_consistent_with_binary_trivial_data)
{
	// Generates data for y = x1 > 5 as decision boundary
	int32_t num_train = 10;
	int32_t num_test = 10;
	int32_t num_trees = 10;

	sg_rand->set_seed(42);
	SGMatrix<float64_t> data_B(1, num_train, false);

	for (auto i = 0; i < num_train; ++i)
	{
		data_B(0, i) = i < 5 ? CMath::random(0, 5) : CMath::random(5, 10);
	}
	CDenseFeatures<float64_t>* features_train =
	    new CDenseFeatures<float64_t>(data_B);

	float64_t labels[] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0};
	SGVector<float64_t> lab(labels, num_train);
	CMulticlassLabels* labels_train = new CMulticlassLabels(lab);

	SGMatrix<float64_t> test_data(1, num_test, false);

	for (auto i = 0; i < num_test; ++i)
	{
		test_data(0, i) = i < 5 ? CMath::random(0, 4) : CMath::random(6, 10);
	}

	CDenseFeatures<float64_t>* features_test =
	    new CDenseFeatures<float64_t>(test_data);

	CRandomForest* c =
	    new CRandomForest(features_train, labels_train, num_trees, 1);
	SGVector<bool> ft = SGVector<bool>(1);
	ft[0] = false;
	c->set_feature_types(ft);

	CMeanRule* mr = new CMeanRule();
	c->set_combination_rule(mr);
	c->train(features_train);

	auto result = c->apply_binary(features_test);
	SGVector<float64_t> res_vector = result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(-1.0, res_vector[0]);
	EXPECT_EQ(-1.0, res_vector[1]);
	EXPECT_EQ(-1.0, res_vector[2]);
	EXPECT_EQ(-1.0, res_vector[3]);
	EXPECT_EQ(-1.0, res_vector[4]);
	EXPECT_EQ(1.0, res_vector[5]);
	EXPECT_EQ(1.0, res_vector[6]);
	EXPECT_EQ(1.0, res_vector[7]);
	EXPECT_EQ(1.0, res_vector[8]);
	EXPECT_EQ(1.0, res_vector[9]);

	EXPECT_NEAR(0.0, values_vector[0], 1e-1);
	EXPECT_NEAR(0.0, values_vector[1], 1e-1);
	EXPECT_NEAR(0.0, values_vector[2], 1e-1);
	EXPECT_NEAR(0.0, values_vector[3], 1e-1);
	EXPECT_NEAR(0.0, values_vector[4], 1e-1);
	EXPECT_NEAR(1.0, values_vector[5], 1e-1);
	EXPECT_NEAR(1.0, values_vector[6], 1e-1);
	EXPECT_NEAR(1.0, values_vector[7], 1e-1);
	EXPECT_NEAR(1.0, values_vector[8], 1e-1);
	EXPECT_NEAR(1.0, values_vector[9], 1e-1);

	SG_UNREF(result);
}