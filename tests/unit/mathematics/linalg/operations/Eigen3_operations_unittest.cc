#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendEigen, SGVector_add)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;

	SGVector<float64_t> A(9);
	SGVector<float64_t> B(9);

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	auto result = add(A, B, alpha, beta);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_add)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float64_t> A(nrows, ncols);
	SGMatrix<float64_t> B(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	auto result = add(A, B, alpha, beta);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_add_in_place)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;

	SGVector<float64_t> A(9), B(9), C(9);

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}

	add(A, B, A, alpha, beta);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_add_in_place)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float64_t> A(nrows, ncols);
	SGMatrix<float64_t> B(nrows, ncols);
	SGMatrix<float64_t> C(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}

	add(A, B, A, alpha, beta);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_dot)
{
	const index_t size = 3;
	SGVector<int32_t> a(size), b(size);
	a.range_fill(0);
	b.range_fill(0);

	auto result = dot(a, b);

	EXPECT_NEAR(result, 5, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_elementwise_product)
{
	const index_t nrows = 3;
	const index_t ncols = 3;
	SGMatrix<float64_t> A(nrows,ncols);
	SGMatrix<float64_t> B(nrows,ncols);
	SGMatrix<float64_t> C(nrows,ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	C = element_prod(A, B);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(A[i]*B[i], C[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_elementwise_product_in_place)
{
	const index_t nrows = 3;
	const index_t ncols = 3;
	SGMatrix<float64_t> A(nrows,ncols);
	SGMatrix<float64_t> B(nrows,ncols);
	SGMatrix<float64_t> C(nrows,ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
		C[i] = i;
	}

	element_prod(A, B, A);
	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(C[i]*B[i], A[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_max)
{
	SGVector<float64_t> A(9);

	float64_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 4};

	for (int32_t i=0; i<9; i++)
		A[i] = a[i];

	EXPECT_NEAR(8, max(A), 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_max)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> A(nrows, ncols);

	float64_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 4};

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = a[i];

	EXPECT_NEAR(8, max(A), 1e-15);
}

TEST(LinalgBackendEigen, SGVector_mean)
{
	const index_t size = 6;
	SGVector<int32_t> vec(size);
	vec.range_fill(0);

	auto result = mean(vec);

	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_mean)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = mean(mat);

	EXPECT_NEAR(result, 2.5, 1E-15);
}

TEST(LinalgBackendEigen, SGVector_range_fill)
{
	const index_t size = 5;
	SGVector<int32_t> vec(size);
	range_fill(vec, 1);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(vec[i], i + 1, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_range_fill)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);
	range_fill(mat, 1);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(mat[i], i + 1, 1E-15);
}

TEST(LinalgBackendEigen, SGVector_scale)
{
	const index_t size = 5;
	const float64_t alpha = 0.3;
	SGVector<float64_t> a(size);
	a.range_fill(0);

	auto result = scale(a, alpha);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * a[i], result[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_scale)
{
	const float64_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> A(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	auto result = scale(A, alpha);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*A[i], result[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_scale_in_place)
{
	const index_t size = 5;
	const float64_t alpha = 0.3;
	SGVector<float64_t> a(size);
	a.range_fill(0);

	scale(a, a, alpha);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * i, a[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_scale_in_place)
{
	const float64_t alpha = 0.3;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float64_t> A(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;

	scale(A, A, alpha);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(alpha*i, A[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_set_const)
{
	const index_t size = 5;
	const float64_t value = 2;
	SGVector<float64_t> a(size);

	set_const(a, value);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_set_const)
{
	const index_t nrows = 2, ncols = 3;
	const float64_t value = 2;
	SGMatrix<float64_t> a(nrows, ncols);

	set_const(a, value);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_NEAR(a[i], value, 1E-15);
}

TEST(LinalgBackendEigen, SGVector_sum)
{
	const index_t size = 10;
	SGVector<int32_t> vec(size);
	vec.range_fill(0);

	auto result = sum(vec);

	EXPECT_NEAR(result, 45, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = sum(mat);

	EXPECT_NEAR(result, 15, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = sum(mat, true);

	EXPECT_NEAR(result, 12, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_block_sum)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = 0; j < n; ++j)
			mat(i, j)=i * 10 + j + 1;

	auto result = sum(linalg::block(mat, 0, 0, 2, 3));
	EXPECT_NEAR(result, 42.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_colwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<int32_t> result = colwise_sum(mat);

	for (index_t j = 0; j < ncols; ++j)
	{
		int32_t sum = 0;
		for (index_t i = 0; i < nrows; ++i)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[j], 1E-15);
	}
}

TEST(LinalgBackendEigen, SGMatrix_colwise_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<int32_t> result = colwise_sum(mat, true);

	EXPECT_NEAR(result[0], 1, 1E-15);
	EXPECT_NEAR(result[1], 2, 1E-15);
	EXPECT_NEAR(result[2], 9, 1E-15);
}


TEST(LinalgBackendEigen, SGMatrix_block_colwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
			mat(i, j) = i * 10 + j + 1;

	auto result = colwise_sum(linalg::block(mat, 0, 0, 2, 3));

	for (index_t j = 0; j < ncols; ++j)
	{
		float64_t sum = 0;
		for (index_t i = 0; i < nrows; ++i)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[j], 1E-15);
	}
}

TEST(LinalgBackendEigen, SGMatrix_rowwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<int32_t> result = rowwise_sum(mat);

	for (index_t i = 0; i < nrows; ++i)
	{
		int32_t sum = 0;
		for (index_t j = 0; j < ncols; ++j)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[i], 1E-15);
	}
}

TEST(LinalgBackendEigen, SGMatrix_rowwise_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<int32_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<int32_t> result = rowwise_sum(mat, true);

	EXPECT_NEAR(result[0], 6, 1E-15);
	EXPECT_NEAR(result[1], 6, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_block_rowwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
			mat(i, j) = i * 10 + j + 1;

	auto result = rowwise_sum(linalg::block(mat, 0, 0, 2, 3));

	for (index_t i = 0; i < nrows; ++i)
	{
		float64_t sum = 0;
		for (index_t j = 0; j < ncols; ++j)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[i], 1E-15);
	}
}
