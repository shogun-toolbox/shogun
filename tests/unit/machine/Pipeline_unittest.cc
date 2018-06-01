#include "features/MockFeatures.h"
#include "labels/MockLabels.h"
#include "machine/MockMachine.h"
#include "transformer/MockTransformer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/machine/Pipeline.h>

using namespace shogun;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::_;
using ::testing::InSequence;

class PipelineTest : public ::testing::Test
{
protected:
	virtual void SetUp()
	{
		ON_CALL(*transformer1, train_require_labels())
		    .WillByDefault(Return(false));
		ON_CALL(*transformer2, train_require_labels())
		    .WillByDefault(Return(true));
	}

	Some<CPipeline> pipeline = some<CPipeline>();
	Some<NiceMock<MockCTransformer>> transformer1 =
	    some<NiceMock<MockCTransformer>>();
	Some<NiceMock<MockCTransformer>> transformer2 =
	    some<NiceMock<MockCTransformer>>();
	Some<NiceMock<MockCMachine>> machine = some<NiceMock<MockCMachine>>();
};

TEST_F(PipelineTest, no_machine)
{
	pipeline->with(transformer1);

	auto features = some<NiceMock<MockCFeatures>>();
	EXPECT_THROW(pipeline->train(features), ShogunException);
}

TEST_F(PipelineTest, wrong_order)
{
	EXPECT_THROW(pipeline->then(machine)->with(transformer1), ShogunException);
}

TEST_F(PipelineTest, multiple_machine)
{
	EXPECT_THROW(pipeline->then(machine)->then(machine), ShogunException);
}

TEST_F(PipelineTest, fit_predict)
{
	auto features = some<NiceMock<MockCFeatures>>();
	auto labels = some<NiceMock<MockCLabels>>();
	pipeline->with(transformer1)->with(transformer2)->then(machine);

	// no labels given
	EXPECT_THROW(pipeline->train(features), ShogunException);

	InSequence s;

	EXPECT_CALL(*transformer1, fit(_));
	EXPECT_CALL(*transformer1, fit(_, _)).Times(0);
	EXPECT_CALL(*transformer1, transform(_, _));
	EXPECT_CALL(*transformer2, fit(_)).Times(0);
	EXPECT_CALL(*transformer2, fit(_, _));
	EXPECT_CALL(*transformer2, transform(_, _));
	EXPECT_CALL(*machine, train_machine(_));

	pipeline->set_labels(labels);
	pipeline->train(features);

	Mock::VerifyAndClearExpectations(transformer1);
	Mock::VerifyAndClearExpectations(transformer2);
	Mock::VerifyAndClearExpectations(machine);

	EXPECT_CALL(*transformer1, transform(_, _));
	EXPECT_CALL(*transformer2, transform(_, _));
	EXPECT_CALL(*machine, apply(_));

	pipeline->apply(features);
}

TEST_F(PipelineTest, get)
{
	EXPECT_THROW(pipeline->get_transformer(0), ShogunException);
	EXPECT_THROW(pipeline->get_machine(), ShogunException);

	pipeline->with(transformer1)->with(transformer2)->then(machine);

	EXPECT_EQ(pipeline->get_transformer(1), transformer2.get());
	EXPECT_EQ(pipeline->get_machine(), machine.get());
}
