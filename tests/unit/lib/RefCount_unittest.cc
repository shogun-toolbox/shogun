#include <shogun/lib/RefCount.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(RefCount,ref_unref)
{
	RefCount r;
	EXPECT_EQ(r.ref_count(), 0);

	r.ref();
	EXPECT_EQ(r.ref_count(), 1);

	r.unref();
	EXPECT_EQ(r.ref_count(), 0);
}

