#include "features/MockFeatures.h"
#include "labels/MockLabels.h"
#include "machine/MockMachine.h"
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/config.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/tree/CARTree.h>
#include "utils/Utils.h"

using namespace shogun;
using ::testing::Return;

class BaggingMachine : public ::testing::Test
{
public:
	CDenseFeatures<float64_t>* features_test;
	CDenseFeatures<float64_t>* features_train;
	CMulticlassLabels* labels_train;

	SGVector<bool> ft;
	virtual void SetUp()
	{
		sg_rand->set_seed(1);
		load_toy_data();
	}

	virtual void TearDown()
	{
		SG_UNREF(features_train);
		SG_UNREF(features_test);
		SG_UNREF(labels_train);
	}

	void load_toy_data()
	{
		SGMatrix<float64_t> weather_data(4, 14);
		SGVector<float64_t> lab(14);

		generate_toy_data_weather(weather_data, lab);

		features_train = new CDenseFeatures<float64_t>(weather_data);
		labels_train = new CMulticlassLabels(lab);

		SGMatrix<float64_t> test(4, 5);
		SGVector<float64_t> test_labels(4);
		generate_toy_data_weather(test, test_labels, false);
		features_test = new CDenseFeatures<float64_t>(test);

		auto feature_types = SGVector<bool>(4);

		feature_types[0] = true;
		feature_types[1] = true;
		feature_types[2] = true;
		feature_types[3] = true;

		ft = feature_types;

		SG_REF(features_train);
		SG_REF(features_test);
		SG_REF(labels_train);
	}
};

/** gmock REV 443 and freebsd doesn't play nicely */
#ifdef FREEBSD
TEST_F(BaggingMachine, DISABLED_mock_train)
#else
TEST_F(BaggingMachine, mock_train)
#endif
{
	using ::testing::NiceMock;
	using ::testing::_;
	using ::testing::InSequence;
	using ::testing::Mock;
	using ::testing::DefaultValue;

	int32_t bag_size = 20;
	int32_t num_bags = 10;

	NiceMock<MockCFeatures> features; features.ref();
	NiceMock<MockCLabels> labels; labels.ref();
	CBaggingMachine* bm = new CBaggingMachine(&features, &labels);
	NiceMock<MockCMachine> mm; mm.ref();
	CMajorityVote* mv = new CMajorityVote();

	bm->parallel->set_num_threads(1);
	bm->set_machine(&mm);
	bm->set_bag_size(bag_size);
	bm->set_num_bags(num_bags);
	bm->set_combination_rule(mv);

	ON_CALL(mm, train_machine(_))
		.WillByDefault(Return(true));

	ON_CALL(features, get_num_vectors())
		.WillByDefault(Return(100));

	{
		InSequence s;
		for (int i = 0; i < num_bags; i++) {
			EXPECT_CALL(mm, clone())
				.Times(1)
				.WillRepeatedly(Return(&mm));

			EXPECT_CALL(mm, train_machine(_))
				.Times(1)
				.WillRepeatedly(Return(true));

			mm.ref();
		}
	}

	bm->train();

	SG_UNREF(bm);
}

TEST_F(BaggingMachine, classify_CART)
{
	CCARTree* cart=new CCARTree();
	CMajorityVote* cv=new CMajorityVote();
	cart->set_feature_types(ft);

	auto c = some<CBaggingMachine>(features_train, labels_train);

	c->parallel->set_num_threads(1);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->train(features_train);

	CMulticlassLabels* result = c->apply_multiclass(features_test);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);

	auto eval = some<CMulticlassAccuracy>();
	EXPECT_NEAR(0.642857,c->get_oob_error(eval),1e-6);

	SG_UNREF(result);
}

TEST_F(BaggingMachine, output_binary)
{
	CCARTree* cart = new CCARTree();
	CMeanRule* cv = new CMeanRule();

	cart->set_feature_types(ft);
	auto c = some<CBaggingMachine>(features_train, labels_train);
	c->parallel->set_num_threads(1);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->train(features_train);

	CBinaryLabels* result = c->apply_binary(features_test);
	SGVector<float64_t> res_vector = result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(1.0, res_vector[0]);
	EXPECT_EQ(-1.0, res_vector[1]);
	EXPECT_EQ(-1.0, res_vector[2]);
	EXPECT_EQ(1.0, res_vector[3]);
	EXPECT_EQ(1.0, res_vector[4]);

	EXPECT_DOUBLE_EQ(1.0, values_vector[0]);
	EXPECT_DOUBLE_EQ(0.3, values_vector[1]);
	EXPECT_DOUBLE_EQ(0.3, values_vector[2]);
	EXPECT_DOUBLE_EQ(1.0, values_vector[3]);
	EXPECT_DOUBLE_EQ(0.7, values_vector[4]);

	SG_UNREF(result);
}

TEST_F(BaggingMachine, output_multiclass_probs_sum_to_one)
{

	CCARTree* cart = new CCARTree();
	CMeanRule* cv = new CMeanRule();

	cart->set_feature_types(ft);
	auto c = some<CBaggingMachine>(features_train, labels_train);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->train(features_train);

	CMulticlassLabels* result = c->apply_multiclass(features_test);

	SGVector<float64_t> res_vector = result->get_labels();
	SGVector<float64_t> values_vector = result->get_values();

	EXPECT_EQ(1.0, res_vector[0]);
	EXPECT_EQ(0.0, res_vector[1]);
	EXPECT_EQ(0.0, res_vector[2]);
	EXPECT_EQ(1.0, res_vector[3]);
	EXPECT_EQ(1.0, res_vector[4]);

	int32_t num_labels = result->get_num_labels();

	for (int32_t i = 0; i < num_labels; ++i)
	{
		SGVector<float64_t> confidences = result->get_multiclass_confidences(i);
		EXPECT_DOUBLE_EQ(1.0, linalg::sum(confidences));
	}

	SG_UNREF(result);
}
