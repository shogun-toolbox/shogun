#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <shogun/base/init.h>

using namespace shogun;
using ::testing::Test;

int main(int argc, char** argv)
{
	::testing::InitGoogleMock(&argc, argv);
	init_shogun();
	int ret = RUN_ALL_TESTS();
	exit_shogun();

	return ret;
}

