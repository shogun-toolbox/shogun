/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2019 Viktor Gal
 */
#include <gtest/gtest.h>

#include <shogun/base/class_list.h>

using namespace shogun;
using namespace std;


TEST(class_list, available_objects)
{
	auto class_list = available_objects();
	EXPECT_TRUE(class_list.find("mock_class") != class_list.end());
	EXPECT_TRUE(class_list.find("another_mock_class") != class_list.end());
}

TEST(class_list, create)
{
	auto mock_class_obj = create("mock_class", PT_SGOBJECT);
	ASSERT_TRUE(mock_class_obj);

	auto another_mock_class_obj = create("another_mock_class", PT_SGOBJECT);
	ASSERT_TRUE(another_mock_class_obj);

	std::string mock_class_name = mock_class_obj->get_name();
	std::string another_mock_class_name = another_mock_class_obj->get_name();

	EXPECT_EQ("MockClass", mock_class_name);
	EXPECT_EQ("AnotherMockClass", another_mock_class_name);
}

