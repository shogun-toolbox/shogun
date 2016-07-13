#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendViennaCL, SGVector_to_gpu_viennacl)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size);
	a.range_fill(0);
	b = to_gpu(a);

	EXPECT_FALSE(a.on_gpu());
	EXPECT_TRUE(b.on_gpu());
}

TEST(LinalgBackendViennaCL, SGMatrix_to_gpu_viennacl)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), b(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
			a[i] = i;
	b = to_gpu(a);

	EXPECT_FALSE(a.on_gpu());
	EXPECT_TRUE(b.on_gpu());
}

TEST(LinalgBackendViennaCL, SGVector_from_gpu_viennacl_on_gpu_flags_check)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size), c(size);
	a.range_fill(0);

	b = to_gpu(a);
	c = from_gpu(b);

	EXPECT_TRUE(b.on_gpu());
	EXPECT_FALSE(c.on_gpu());
}

TEST(LinalgBackendViennaCL, SGVector_from_gpu_viennacl_values_check)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size), c(size);
	a.range_fill(0);

	b = to_gpu(a);
	c = from_gpu(b);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(i, c[i], 1E-15);
}

TEST(LinalgBackendViennaCL, SGMatrix_from_gpu_viennacl_values_check)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols), b(nrows, ncols), c;

	for (index_t i = 0; i < nrows * ncols; ++i)
			a[i] = i;

	b = to_gpu(a);
	c = from_gpu(b);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(a[i], c[i], 1E-15);
}

#endif // HAVE_VIENNACL
