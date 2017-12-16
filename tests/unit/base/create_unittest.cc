#include <shogun/base/class_list.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/Machine.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(CreateObject,create_wrong_name)
{
    EXPECT_THROW(create_object<CKernel>("GoussianKernel"), ShogunException);
}

TEST(CreateObject,create_wrong_type)
{
    EXPECT_THROW(create_object<CMachine>("GaussianKernel"), ShogunException);
}

TEST(CreateObject,create_wrong_type_wrong_name)
{
    EXPECT_THROW(create_object<CMachine>("GoussianKernel"), ShogunException);
}

TEST(CreateObject,create)
{
    auto* obj = create_object<CKernel>("GaussianKernel");
    EXPECT_TRUE(obj != nullptr);
    EXPECT_TRUE(dynamic_cast<CKernel*>(obj) != nullptr);
    delete obj;
}

TEST(CreateObject,create_kernel)
{
    auto* obj = kernel("GaussianKernel");
    EXPECT_TRUE(obj != nullptr);
    EXPECT_TRUE(dynamic_cast<CKernel*>(obj) != nullptr);
    delete obj;
}
