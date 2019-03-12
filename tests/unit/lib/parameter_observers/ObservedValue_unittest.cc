/*
* Written (W) 2019 Giovanni De Toni
*/

#include "sg_gtest_utilities.h"
#include <shogun/lib/config.h>
#include <shogun/lib/parameter_observers/ObservedValueTemplated.h>

using namespace shogun;

TEST(ObservedValue, set_correct)
{
	auto obs = ObservedValue::make_observation(1, "test", 42);
	EXPECT_EQ(obs->get<int64_t>("step"), 1);
	EXPECT_EQ(obs->get<std::string>("name"), "test");
	EXPECT_EQ(obs->get<int32_t>("value"), 42);
};