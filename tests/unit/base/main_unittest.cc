#include <iostream>
#include <gtest/gtest.h>
#include <shogun/base/init.h>

using namespace shogun;
using ::testing::Test;
using ::testing::InitGoogleTest;

int main(int argc, char** argv)
{
	InitGoogleTest(&argc, argv);
	init_shogun();
	int ret = RUN_ALL_TESTS();
	exit_shogun();

	return ret;
}

