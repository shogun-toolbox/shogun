/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#include <gtest/gtest.h>
#include <shogun/optimization/Space.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/ConstKernel.h>


using namespace shogun;
TEST(Space, constructor) 
{
    auto guassian_kernel = std::make_shared<GaussianKernel>();
    auto const_kernel = std::make_shared<ConstKernel>();
   
    Space s{{
        Dimension("C1", {1.0, 2.0}),
        Dimension("C2", {1.0, 4.0}),
        DimensionWithSGObject({"kernel", {
            {guassian_kernel, {"width", {1.0 ,2.0}}},
            {const_kernel, {"const_value", {1.0, 2.0}}}}})}};
}