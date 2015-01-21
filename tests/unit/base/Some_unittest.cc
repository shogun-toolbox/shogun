#include <shogun/base/some.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Some,basic)
{
    // raw pointer to the kernel
    CKernel* raw = NULL;
    // local scope to create kernel
    {
        auto kernel = some<CGaussianKernel>();
        raw = kernel.get();
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
