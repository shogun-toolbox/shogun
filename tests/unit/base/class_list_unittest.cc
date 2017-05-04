/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Sanuj Sharma
 */

#include <shogun/base/SGObject.h>
#include <shogun/base/class_list.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(class_list, create_gaussian_kernel)
{
    const char* class_name = "GaussianKernel";
    auto gk = create<CKernel>(class_name);
    auto another_gk = new CGaussianKernel();

    EXPECT_EQ(strcmp(gk->get_name(), class_name), 0);
    EXPECT_TRUE(gk->equals(another_gk));
    delete another_gk;
}
