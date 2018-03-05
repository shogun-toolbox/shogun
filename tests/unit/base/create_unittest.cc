#include <shogun/base/class_list.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/Machine.h>
#include <shogun/features/DenseFeatures.h>

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

TEST(CreateObject,create_wrong_ptype)
{
    EXPECT_THROW(create_object<CMachine>("GaussianKernel", PT_FLOAT64), ShogunException);
}

TEST(CreateObject,create_wrong_ptype2)
{
    EXPECT_THROW(create_object<CMachine>("DenseFeatures"), ShogunException);
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
    EXPECT_EQ(obj->get_generic(), PT_NOT_GENERIC);
    delete obj;
}

TEST(CreateObject,create_with_ptype)
{
    auto* obj = create_object<CDenseFeatures<float64_t>>("DenseFeatures", PT_FLOAT64);
    EXPECT_TRUE(obj != nullptr);
    EXPECT_TRUE(dynamic_cast<CDenseFeatures<float64_t>*>(obj) != nullptr);
    EXPECT_EQ(obj->get_generic(), PT_FLOAT64);
    delete obj;
}
