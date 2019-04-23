#include "features/MockFeatures.h"
#include "labels/MockLabels.h"
#include "machine/MockMachine.h"
#include "utils/Utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/ensemble/MeanRule.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/config.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/multiclass/tree/CARTree.h>

using namespace shogun;
using ::testing::Return;

class BaggingMachineTest : public ::testing::Test
{
public:
	std::shared_ptr<DenseFeatures<float64_t>> features_test;
	std::shared_ptr<DenseFeatures<float64_t>> features_train;
	std::shared_ptr<MulticlassLabels> labels_train;

	SGVector<bool> ft;
	virtual void SetUp()
	{
		load_toy_data();
	}

	virtual void TearDown()
	{
	}

	void load_toy_data()
	{
		SGMatrix<float64_t> weather_data(4, 14);
		SGVector<float64_t> lab(14);

		generate_toy_data_weather(weather_data, lab);

		features_train = std::make_shared<DenseFeatures<float64_t>>(weather_data);
		labels_train = std::make_shared<MulticlassLabels>(lab);

		SGMatrix<float64_t> test(4, 5);
		SGVector<float64_t> test_labels(4);
		generate_toy_data_weather(test, test_labels, false);
		features_test = std::make_shared<DenseFeatures<float64_t>>(test);

		auto feature_types = SGVector<bool>(4);

		feature_types[0] = true;
		feature_types[1] = true;
		feature_types[2] = true;
		feature_types[3] = true;

		ft = feature_types;
	}
};

/** gmock REV 443 and freebsd doesn't play nicely */
#ifdef FREEBSD
TEST_F(BaggingMachineTest, DISABLED_mock_train)
#else
TEST_F(BaggingMachineTest, mock_train)
#endif
{
	using ::testing::NiceMock;
	using ::testing::_;
	using ::testing::InSequence;
	using ::testing::Mock;
	using ::testing::DefaultValue;

	int32_t bag_size = 20;
	int32_t num_bags = 10;
	int32_t seed = 25;

	auto features = std::make_shared<NiceMock<MockFeatures>>();
	auto labels = std::make_shared<NiceMock<MockLabels>>();
	auto bm = std::make_shared<BaggingMachine>(features, labels);
	auto mm = std::make_shared<NiceMock<MockMachine>>();
	auto mv = std::make_shared<MajorityVote>();

	env()->set_num_threads(1);
	bm->set_machine(mm);
	bm->set_bag_size(bag_size);
	bm->set_num_bags(num_bags);
	bm->set_combination_rule(mv);
	bm->put("seed", seed);

	ON_CALL(*mm, train_machine(_))
		.WillByDefault(Return(true));

	ON_CALL(*features, get_num_vectors())
		.WillByDefault(Return(100));

	{
		InSequence s;
		for (int i = 0; i < num_bags; i++) {
			EXPECT_CALL(*mm, clone(ParameterProperties::ALL))
				.Times(1)
				.WillRepeatedly(Return(mm));

			EXPECT_CALL(*mm, train_machine(_))
				.Times(1)
				.WillRepeatedly(Return(true));
		}
	}

	bm->train();
}

TEST_F(BaggingMachineTest, classify_CART)
{
	int32_t seed = 555;
	auto cart=std::make_shared<CARTree>();
	auto cv=std::make_shared<MajorityVote>();
	cart->set_feature_types(ft);

	auto c = std::make_shared<BaggingMachine>(features_train, labels_train);

	env()->set_num_threads(1);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->put("seed", seed);
	c->train(features_train);

	auto result = c->apply_multiclass(features_test);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);

	auto eval = std::make_shared<MulticlassAccuracy>();
	c->put(BaggingMachine::kOobEvaluationMetric, eval);
	EXPECT_NEAR(0.642857,c->get<float64_t>(BaggingMachine::kOobError),1e-6);

}

TEST_F(BaggingMachineTest, output_binary)
{
	int32_t seed = -1051963731;
	auto cart = std::make_shared<CARTree>();
	auto cv = std::make_shared<MeanRule>();

	cart->set_feature_types(ft);
	auto c = std::make_shared<BaggingMachine>(features_train, labels_train);
	env()->set_num_threads(1);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->put("seed", seed);
	c->train(features_train);

	auto result = c->apply_binary(features_test);
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


}

TEST_F(BaggingMachineTest, output_multiclass_probs_sum_to_one)
{
	int32_t seed = 24;

	auto cart = std::make_shared<CARTree>();
	auto cv = std::make_shared<MajorityVote>();

	cart->set_feature_types(ft);
	auto c = std::make_shared<BaggingMachine>(features_train, labels_train);
	c->set_machine(cart);
	c->set_bag_size(14);
	c->set_num_bags(10);
	c->set_combination_rule(cv);
	c->put("seed", seed);
	c->train(features_train);

	auto result = c->apply_multiclass(features_test);

	SGVector<float64_t> res_vector = result->get_labels();

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


}
