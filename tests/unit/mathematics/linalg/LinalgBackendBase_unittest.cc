#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendBase, SGVector_to_gpu_without_gpu_backend)
{
	sg_linalg->set_gpu_backend(nullptr);

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size), c;
	a.range_fill(0);
	to_gpu(a, b);
	from_gpu(b, c);

	EXPECT_FALSE(a.on_gpu());
	EXPECT_FALSE(b.on_gpu());
	EXPECT_FALSE(c.on_gpu());
	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], c[i], 1E-15);
}

TEST(LinalgBackendBase, SGMatrix_to_gpu_without_gpu_backend)
{
	sg_linalg->set_gpu_backend(nullptr);

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), b(nrows, ncols), c;

	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	to_gpu(a, b);;
	from_gpu(b, c);

	EXPECT_FALSE(a.on_gpu());
	EXPECT_FALSE(b.on_gpu());
	EXPECT_FALSE(c.on_gpu());
	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(a[i], c[i], 1E-15);
}
