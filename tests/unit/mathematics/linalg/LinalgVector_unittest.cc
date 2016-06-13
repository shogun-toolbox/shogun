#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <memory>
#include <gtest/gtest.h>
#include <shogun/mathematics/linalg/linalgVector.h>

using namespace shogun;

#ifdef HAVE_CXX11

TEST(LinalgVector, deepcopy_constructor)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	LinalgVector<int32_t> a_vec(a);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], a_vec[i], 1E-15);

}

#endif
