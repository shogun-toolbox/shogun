#include <shogun/base/interface.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11

using namespace shogun;

TEST(Interface,create_object)
{
    auto kernel = object("GaussianKernel");
    EXPECT_TRUE(static_cast<CSGObject*>(kernel));
    EXPECT_STREQ(kernel->get_name(), "GaussianKernel");
}

TEST(Interface,create_unknown_object)
{
    auto foo = object("FooFooFooBarBarBar");
    EXPECT_FALSE(static_cast<CSGObject*>(foo));
}

#endif
