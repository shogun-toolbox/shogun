#include <gtest/gtest.h>

#include <shogun/base/class_list.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/machine/KernelMachine.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

TEST(CreateObject,create_wrong_name)
{
    EXPECT_THROW(create_object<Kernel>("GoussianKernel"), ShogunException);
}

TEST(CreateObject,create_wrong_type)
{
    EXPECT_THROW(create_object<Machine>("GaussianKernel"), ShogunException);
}

TEST(CreateObject, create_wrong_ptype)
{
	EXPECT_THROW(
	    create_object<Machine>("GaussianKernel", PT_FLOAT64), ShogunException);
}

TEST(CreateObject, create_wrong_ptype2)
{
	EXPECT_THROW(create_object<Machine>("DenseFeatures"), ShogunException);
}

TEST(CreateObject,create_wrong_type_wrong_name)
{
    EXPECT_THROW(create_object<Machine>("GoussianKernel"), ShogunException);
}

TEST(CreateObject,create)
{
    auto obj = create_object<Kernel>("GaussianKernel");
    EXPECT_TRUE(obj != nullptr);
    EXPECT_TRUE(obj->as<Kernel>() != nullptr);
	EXPECT_EQ(obj->get_generic(), PT_NOT_GENERIC);
}

TEST(CreateObject, create_with_ptype)
{
	auto obj =
	    create_object<DenseFeatures<float64_t>>("DenseFeatures", PT_FLOAT64);
	EXPECT_TRUE(obj != nullptr);
	EXPECT_TRUE(obj->as<DenseFeatures<float64_t>>() != nullptr);
	EXPECT_EQ(obj->get_generic(), PT_FLOAT64);
}
