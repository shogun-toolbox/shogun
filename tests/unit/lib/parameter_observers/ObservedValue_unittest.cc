/*
* Written (W) 2019 Giovanni De Toni
*/

#include "sg_gtest_utilities.h"
#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>

using namespace shogun;

TEST(ObservedValue, set_correct)
{
	auto obs = some<>(1, "test", "test description", 42);
	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("value"), 42);
};

TEST(ObservedValue, set_correct_properties)
{
	AnyParameterProperties any_prop("test description",
										   ParameterProperties::MODEL);
	auto obs = ObservedValue::make_observation(1, "test", 42, any_prop);

	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("value"), 42);
	EXPECT_TRUE(obs->get_params().find("value")->second->get_properties().compare_mask(ParameterProperties::MODEL));
};

TEST(ObservedValue, set_correct_parameter)
{
	int32_t p = 42;
	AnyParameter param(make_any(p),
					   AnyParameterProperties("test description", ParameterProperties::MODEL));

	auto obs = ObservedValue::make_observation<int32_t>(1, "test", param);
	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("value"), 42);
};