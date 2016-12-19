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

TEST(LinalgBackendViennaCL, SGVector_clone)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	auto b = to_gpu(a);
	ASSERT_EQ(a.size(), b.size());

	auto d = b.clone();
	ASSERT_EQ(b.size(), d.size());

	auto c = from_gpu(d);
	ASSERT_EQ(c.size(), d.size());

	for (index_t i = 0; i < size; ++i)
		EXPECT_EQ(a[i], c[i]);
}

TEST(LinalgBackendViennaCL, SGMatrix_clone)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> a(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		a[i] = i;

	auto b = to_gpu(a);
	ASSERT_EQ(a.num_cols, b.num_cols);
	ASSERT_EQ(a.num_rows, b.num_rows);

	auto d = b.clone();
	ASSERT_EQ(b.num_cols, d.num_cols);
	ASSERT_EQ(b.num_rows, d.num_rows);

	auto c = from_gpu(d);
	ASSERT_EQ(c.num_cols, d.num_cols);
	ASSERT_EQ(c.num_rows, d.num_rows);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_EQ(a[i], c[i]);
}

#endif // HAVE_VIENNACL
