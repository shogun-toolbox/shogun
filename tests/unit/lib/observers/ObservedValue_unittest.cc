/*
* Written (W) 2019 Giovanni De Toni
*/

#include "sg_gtest_utilities.h"
#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>

using namespace shogun;

TEST(ObservedValue, set_correct)
{
	auto obs = std::make_shared<ObservedValueTemplated<int32_t>>(
	    1, "test", "test description", 42);
	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("test"), 42);
};

TEST(ObservedValue, set_correct_parameter)
{
	int32_t p = 42;
	AnyParameterProperties prop("test description", ParameterProperties::MODEL);

	auto obs = std::make_shared<ObservedValueTemplated<int32_t>>(1, "test", p, prop);
	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("test"), 42);
	EXPECT_TRUE(
	    obs->get_params()
	        .find("test")
	        ->second->get_properties()
	        .get_description() == prop.get_description());
	EXPECT_TRUE(
	    obs->get_params().find("test")->second->get_properties().compare_mask(
	        ParameterProperties::MODEL));
};
