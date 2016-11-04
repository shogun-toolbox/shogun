#include <shogun/base/some.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11
#ifdef USE_REFERENCE_COUNTING
using namespace shogun;

TEST(Some,basic)
{
    // raw pointer to the kernel
    CKernel* raw = NULL;
    // local scope to create kernel
    {
        auto kernel = some<CGaussianKernel>();
        EXPECT_EQ(1, kernel->ref_count());
        EXPECT_EQ(1, kernel->ref_count());
        raw = kernel;
        SG_REF(raw);
        EXPECT_TRUE(kernel->equals(raw));

        // reference is held
        EXPECT_EQ(2, kernel->ref_count());
    }
    EXPECT_TRUE(raw);
    // last references now
    EXPECT_EQ(1, raw->ref_count());
    SG_UNREF(raw);
}

TEST(Some,reassignment)
{
    auto kernel = some<CGaussianKernel>();
    CGaussianKernel* raw = new CGaussianKernel();
    EXPECT_EQ(1, kernel->ref_count());
    EXPECT_EQ(0, raw->ref_count());
    kernel = raw;
    EXPECT_TRUE(kernel->equals(raw));
    EXPECT_EQ(1, kernel->ref_count());
}

TEST(Some,self_assignment)
{
    auto kernel = some<CGaussianKernel>();
    kernel = kernel;
    EXPECT_EQ(1, kernel->ref_count());
}

TEST(Some,get)
{
    auto kernel = some<CGaussianKernel>();
    CGaussianKernel* raw = kernel;
    SG_REF(raw);
    EXPECT_TRUE(kernel->equals(raw));
    EXPECT_EQ(2, raw->ref_count());
    SG_UNREF(raw);
    EXPECT_EQ(1, raw->ref_count());
}
#endif
#endif
