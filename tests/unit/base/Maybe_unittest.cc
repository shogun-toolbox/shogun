#include <shogun/base/maybe.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

/** Test creating absent Maybe from Nothing with
 * specified reason.
 *
 * 1. Create an instance of Maybe to represent absent
 *    value with specified reason.
 * 2. Check that the value is absent.
 * 3. Check that the instance of Maybe evaluates to false.
 */
TEST(Maybe,absent_with_reason)
{
    const char* reason = "Just no kernel";
    Maybe<CGaussianKernel> k = Nothing(reason);
    EXPECT_TRUE(k.is_absent());
    EXPECT_FALSE(k);
}

/** Test creating absent Maybe from Nothing with no
 * specified reason.
 *
 * 1. Create an instance of Maybe to represent absent
 *    value with no reason.
 * 2. Check that the value is absent.
 * 3. Check that the instance of Maybe evaluates to false.
 */
TEST(Maybe,absent_with_no_reason)
{
    Maybe<CGaussianKernel> k = Nothing();
    EXPECT_TRUE(k.is_absent());
    EXPECT_FALSE(k);
}

/** Test creating Maybe with value.
 *
 * 1. Create an instance of Maybe with created kernel.
 * 2. Check that the instance of Maybe evaluates to true.
 * 3. Call equals of kernel on itself and assert it is true.
 *
 * Note: pointer semantics is used to avoid double frees.
 *
 */
TEST(Maybe,present)
{
    Maybe<CGaussianKernel*> k = Just(new CGaussianKernel());
    EXPECT_TRUE(k);
    EXPECT_TRUE(k.value()->equals(k.value()));
    SG_UNREF(k.value());
}

/** Test using alternative for absent value.
 *
 * 1. Create default value for kernel.
 * 2. Create an instance of Maybe with absent value.
 * 3. Check that the instance of Maybe evaluates to false.
 * 4. Obtain kernel with alternative of default kernel.
 * 5. Check that the kernel from the step 4. equals to the
 *    default one.
 *
 * Note: pointer semantics is used to avoid double frees.
 *
 */
TEST(Maybe,value_or)
{
    CGaussianKernel* default_kernel = new CGaussianKernel();
    Maybe<CGaussianKernel*> k = Nothing();
    EXPECT_FALSE(k);
    CGaussianKernel* maybe_kernel = k.value_or(default_kernel);
    EXPECT_TRUE(maybe_kernel->equals(default_kernel));
    SG_UNREF(default_kernel);
}
