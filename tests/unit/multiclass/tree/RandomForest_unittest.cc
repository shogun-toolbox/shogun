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

#include <gtest/gtest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/RandomForest.h>
#include <stdio.h>

using namespace shogun;

#define sunny 1.
#define overcast 2.
#define rain 3.

#define hot 1.
#define mild 2.
#define cool 3.

#define high 1.
#define normal 2.

#define weak 1.
#define strong 2.

void generate_toy_data_weather(
    SGMatrix<float64_t>& data, SGVector<float64_t>& lab)
{
	//vector = [Outlook Temperature Humidity Wind]
	data(0,0)=sunny;
	data(1,0)=hot;
	data(2,0)=high;
	data(3,0)=weak;

	data(0,1)=sunny;
	data(1,1)=hot;
	data(2,1)=high;
	data(3,1)=strong;

	data(0,2)=overcast;
	data(1,2)=hot;
	data(2,2)=high;
	data(3,2)=weak;

	data(0,3)=rain;
	data(1,3)=mild;
	data(2,3)=high;
	data(3,3)=weak;

	data(0,4)=rain;
	data(1,4)=cool;
	data(2,4)=normal;
	data(3,4)=weak;

	data(0,5)=rain;
	data(1,5)=cool;
	data(2,5)=normal;
	data(3,5)=strong;

	data(0,6)=overcast;
	data(1,6)=cool;
	data(2,6)=normal;
	data(3,6)=strong;

	data(0,7)=sunny;
	data(1,7)=mild;
	data(2,7)=high;
	data(3,7)=weak;

	data(0,8)=sunny;
	data(1,8)=cool;
	data(2,8)=normal;
	data(3,8)=weak;

	data(0,9)=rain;
	data(1,9)=mild;
	data(2,9)=normal;
	data(3,9)=weak;

	data(0,10)=sunny;
	data(1,10)=mild;
	data(2,10)=normal;
	data(3,10)=strong;

	data(0,11)=overcast;
	data(1,11)=mild;
	data(2,11)=high;
	data(3,11)=strong;

	data(0,12)=overcast;
	data(1,12)=hot;
	data(2,12)=normal;
	data(3,12)=weak;

	data(0,13)=rain;
	data(1,13)=mild;
	data(2,13)=high;
	data(3,13)=strong;

	lab[0]=0.0;
	lab[1]=0.0;
	lab[2]=1.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=0.0;
	lab[6]=1.0;
	lab[7]=0.0;
	lab[8]=1.0;
	lab[9]=1.0;
	lab[10]=1.0;
	lab[11]=1.0;
	lab[12]=1.0;
	lab[13]=0.0;
}

TEST(RandomForest,classify_nominal_test)
{
	sg_rand->set_seed(1);

	SGMatrix<float64_t> data(4,14);
	SGVector<float64_t> lab(14);

	generate_toy_data_weather(data, lab);

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	SGVector<bool> ft=SGVector<bool>(4);
	ft[0]=true;
	ft[1]=true;
	ft[2]=true;
	ft[3]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CRandomForest* c=new CRandomForest(feats, labels, 100,2);
	c->set_feature_types(ft);
	CMajorityVote* mv = new CMajorityVote();
	c->set_combination_rule(mv);
	c->parallel->set_num_threads(1);
	c->train(feats);

	SGMatrix<float64_t> test(4,5);
	test(0,0)=overcast;
	test(0,1)=rain;
	test(0,2)=sunny;
	test(0,3)=rain;
	test(0,4)=sunny;

	test(1,0)=hot;
	test(1,1)=cool;
	test(1,2)=mild;
	test(1,3)=mild;
	test(1,4)=hot;

	test(2,0)=normal;
	test(2,1)=high;
	test(2,2)=high;
	test(2,3)=normal;
	test(2,4)=normal;

	test(3,0)=strong;
	test(3,1)=strong;
	test(3,2)=weak;
	test(3,3)=weak;
	test(3,4)=strong;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(test);
	CMulticlassLabels* result=(CMulticlassLabels*) c->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);

	CMulticlassAccuracy* eval=new CMulticlassAccuracy();
	EXPECT_NEAR(0.642857,c->get_oob_error(eval),1e-6);

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(eval);
}

TEST(RandomForest,classify_non_nominal_test)
{
	sg_rand->set_seed(1);

	SGMatrix<float64_t> data(4,14);
	SGVector<float64_t> lab(14);

	generate_toy_data_weather(data, lab);

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	SGVector<bool> ft=SGVector<bool>(4);
	ft[0]=false;
	ft[1]=false;
	ft[2]=false;
	ft[3]=false;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CRandomForest* c=new CRandomForest(feats, labels, 100,2);
	c->set_feature_types(ft);
	CMajorityVote* mv = new CMajorityVote();
	c->set_combination_rule(mv);
	c->parallel->set_num_threads(1);
	c->train(feats);

	SGMatrix<float64_t> test(4,5);
	test(0,0)=overcast;
	test(0,1)=rain;
	test(0,2)=sunny;
	test(0,3)=rain;
	test(0,4)=sunny;

	test(1,0)=hot;
	test(1,1)=cool;
	test(1,2)=mild;
	test(1,3)=mild;
	test(1,4)=hot;

	test(2,0)=normal;
	test(2,1)=high;
	test(2,2)=high;
	test(2,3)=normal;
	test(2,4)=normal;

	test(3,0)=strong;
	test(3,1)=strong;
	test(3,2)=weak;
	test(3,3)=weak;
	test(3,4)=strong;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(test);
	CMulticlassLabels* result=(CMulticlassLabels*) c->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);

	CMulticlassAccuracy* eval=new CMulticlassAccuracy();
	EXPECT_NEAR(0.714285,c->get_oob_error(eval),1e-6);

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(eval);
}

TEST(RandomForest, score_compare_sklearn_toydata)
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

TEST(RandomForest, score_consistent_with_binary_trivial_data)
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

	CRandomForest* c = new CRandomForest(features_train, labels_train, num_trees, 1);
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

TEST(RandomForest, score_random_labels)
{
	// Generates data with random labels where trees don't all agree
	sg_rand->set_seed(42);

	int32_t num_train = 1000;
	int32_t num_test = 10;
	int32_t num_trees = 10;
	SGMatrix<float64_t> data_B(2, num_train, false);

	for (auto i = 0; i < num_train; ++i)
	{
		data_B(0, i) = CMath::randn_double();
		data_B(1, i) = CMath::randn_double();
	}
	CDenseFeatures<float64_t>* features_train =
	    new CDenseFeatures<float64_t>(data_B);

	SGVector<float64_t> lab(num_train);
	for (auto i = 0; i < num_train; ++i)
	{
		lab[i] = CMath::random(0, 1);
	}

	CMulticlassLabels* labels_train = new CMulticlassLabels(lab);

	SGMatrix<float64_t> test_data(2, num_test, false);

	for (auto i = 0; i < num_test; ++i)
	{
		test_data(0, i) = CMath::randn_double();
		test_data(1, i) = CMath::randn_double();
	}

	CDenseFeatures<float64_t>* features_test =
	    new CDenseFeatures<float64_t>(test_data);

	CRandomForest* c = new CRandomForest(features_train, labels_train, num_trees, 2);
	SGVector<bool> ft = SGVector<bool>(2);
	ft[0] = false;
	ft[1] = false;
	c->set_feature_types(ft);

	CMeanRule* mr = new CMeanRule();
	c->set_combination_rule(mr);
	c->train(features_train);

	auto result = c->apply_binary(features_test);
	SGVector<float64_t> res_vector = result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_NEAR(0.5, values_vector[0], 5e-1);
	EXPECT_NEAR(0.5, values_vector[1], 5e-1);
	EXPECT_NEAR(0.5, values_vector[2], 5e-1);
	EXPECT_NEAR(0.5, values_vector[3], 5e-1);
	EXPECT_NEAR(0.5, values_vector[4], 5e-1);
	EXPECT_NEAR(0.5, values_vector[5], 5e-1);
	EXPECT_NEAR(0.5, values_vector[6], 5e-1);
	EXPECT_NEAR(0.5, values_vector[7], 5e-1);
	EXPECT_NEAR(0.5, values_vector[8], 5e-1);
	EXPECT_NEAR(0.5, values_vector[9], 5e-1);

	SG_UNREF(result);

}
