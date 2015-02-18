#include <shogun/base/maybe.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11
using namespace shogun;

TEST(Maybe,absent)
{
	const char* reason = "Just no kernel";
	Maybe<CGaussianKernel> k = Maybe<CGaussianKernel>::nope(reason);
	EXPECT_FALSE(k);
}

TEST(Maybe,present)
{
	Maybe<CGaussianKernel> k = Maybe<CGaussianKernel>(CGaussianKernel());
	EXPECT_TRUE(k);
	k->get_name();
}
#endif
