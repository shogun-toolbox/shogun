#include <gtest/gtest.h>

#include <shogun/base/range.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>

using namespace shogun;
using namespace linalg;
using namespace Eigen;

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

TEST(LinalgBackendEigen, SGVector_add_col_vec_allocated)
{
	const float64_t alpha = 0.7;
	const float64_t beta = -1.2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<float64_t> A(nrows, ncols);
	SGVector<float64_t> b(nrows);
	SGVector<float64_t> result(nrows);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 0.5*i;

	add_col_vec(A, col, b, result, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(result[i], alpha*A.get_element(i, col)+beta*b[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_add_col_vec_in_place)
{
	const float64_t alpha = 0.6;
	const float64_t beta = -1.3;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<float64_t> A(nrows, ncols);
	SGVector<float64_t> b(nrows);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 0.5*i;

	add_col_vec(A, col, b, b, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(b[i], alpha*A.get_element(i, col)+beta*0.5*i, 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_add_col_vec_allocated)
{
	const float64_t alpha = 0.8;
	const float64_t beta = -1.4;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<float64_t> A(nrows, ncols);
	SGVector<float64_t> b(nrows);
	SGMatrix<float64_t> result(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 0.5*i;

	add_col_vec(A, col, b, result, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    result.get_element(i, col),
		    alpha * A.get_element(i, col) + beta * b[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_add_col_vec_in_place)
{
	const float64_t alpha = 0.9;
	const float64_t beta = -1.7;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<float64_t> A(nrows, ncols);
	SGVector<float64_t> b(nrows);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 0.5*i;

	add_col_vec(A, col, b, A, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
		{
			float64_t a = i+j*nrows;
			if (j == col)
				EXPECT_NEAR(A.get_element(i, j), alpha*a+beta*b[i], 1e-15);
			else
				EXPECT_EQ(A.get_element(i,j), a);
		}
}

TEST(LinalgBackendEigen, add_diag)
{
	SGMatrix<float64_t> A1(2, 3);
	SGVector<float64_t> b1(2);

	A1(0, 0) = 1;
	A1(0, 1) = 2;
	A1(0, 2) = 3;
	A1(1, 0) = 4;
	A1(1, 1) = 5;
	A1(1, 2) = 6;

	b1[0] = 1;
	b1[1] = 2;

	add_diag(A1, b1, 0.5, 2.0);

	EXPECT_NEAR(A1(0, 0), 2.5, 1e-15);
	EXPECT_NEAR(A1(0, 1), 2, 1e-15);
	EXPECT_NEAR(A1(0, 2), 3, 1e-15);
	EXPECT_NEAR(A1(1, 0), 4, 1e-15);
	EXPECT_NEAR(A1(1, 1), 6.5, 1e-15);
	EXPECT_NEAR(A1(1, 2), 6, 1e-15);

	// test error cases
	SGMatrix<float64_t> A2(2, 2);
	SGVector<float64_t> b2(3);
	SGMatrix<float64_t> A3;
	SGVector<float64_t> b3;
	EXPECT_THROW(add_diag(A2, b2), ShogunException);
	EXPECT_THROW(add_diag(A2, b3), ShogunException);
	EXPECT_THROW(add_diag(A3, b2), ShogunException);
	EXPECT_THROW(add_diag(A3, b3), ShogunException);
}

TEST(LinalgBackendEigen, add_vector)
{
	const float64_t alpha = 0.7;
	const float64_t beta = -1.2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<float64_t> A(nrows, ncols);
	SGMatrix<float64_t> result(nrows, ncols);
	SGVector<float64_t> b(nrows);

	for (index_t i = 0; i < nrows; ++i)
		b[i] = 0.5 * i;
	for (index_t j = 0; j < ncols; ++j)
		for (index_t i = 0; i < nrows; ++i)
		{
			A(i, j) = i + j * ncols;
			result(i, j) = alpha * A(i, j) + beta * b[i];
		}

	add_vector(A, b, A, alpha, beta);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(A[i], result[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_add_scalar)
{
	const index_t n = 4;
	float64_t s = -0.3;

	SGVector<float64_t> a(n);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGVector<float64_t> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_add_scalar)
{
	const index_t r = 4, c = 3;
	float64_t s = 0.4;

	SGMatrix<float64_t> a(r, c);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGMatrix<float64_t> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, 1e-15);
}

TEST(LinalgBackendEigen, center_matrix)
{
	const index_t n = 3;
	float64_t data[] = {0.8192343,  0.13191962, 0.50888604,
	                    0.16857468, 0.24107738, 0.89455301,
	                    0.40657379, 0.07902286, 0.24319651};
	float64_t result[] = {0.25587541,  -0.1173183, -0.13855711,
	                      -0.34283925, 0.04378442, 0.29905482,
	                      0.08696383,  0.07353387, -0.16049771};

	SGMatrix<float64_t> m(data, n, n, false);

	center_matrix(m);

	for (index_t i = 0; i < (index_t)m.size(); ++i)
		EXPECT_NEAR(m[i], result[i], 1e-8);
}

TEST(LinalgBackendEigen, SGMatrix_cholesky_llt_lower)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	//lower triangular cholesky decomposition
	SGMatrix<float64_t> L = cholesky_factor(m);

	Map<MatrixXd> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<MatrixXd> map_L(L.matrix,L.num_rows,L.num_cols);
	EXPECT_NEAR((map_A-map_L*map_L.transpose()).norm(),
		0.0, 1E-15);
	EXPECT_EQ(m.num_rows, L.num_rows);
	EXPECT_EQ(m.num_cols, L.num_cols);
}

TEST(LinalgBackendEigen, SGMatrix_cholesky_llt_upper)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	//upper triangular cholesky decomposition
	SGMatrix<float64_t> U = cholesky_factor(m,false);

	Map<MatrixXd> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<MatrixXd> map_U(U.matrix,U.num_rows,U.num_cols);
	EXPECT_NEAR((map_A-map_U.transpose()*map_U).norm(),
		0.0, 1E-15);
	EXPECT_EQ(m.num_rows, U.num_rows);
	EXPECT_EQ(m.num_cols, U.num_cols);
}

TEST(LinalgBackendEigen, SGMatrix_cholesky_ldlt_lower)
{
	const index_t size = 3;
	SGMatrix<float64_t> m(size, size);
	m(0, 0) = 0.0;
	m(0, 1) = 0.0;
	m(0, 2) = 0.0;
	m(1, 0) = 0.0;
	m(1, 1) = 1.0;
	m(1, 2) = 2.0;
	m(2, 0) = 0.0;
	m(2, 1) = 2.0;
	m(2, 2) = 3.0;

	SGMatrix<float64_t> L(size, size);
	SGVector<float64_t> d(size);
	SGVector<index_t> p(size);

	linalg::ldlt_factor(m, L, d, p);

	EXPECT_NEAR(d[0], 3.0, 1e-15);
	EXPECT_NEAR(d[1], -0.333333333333333, 1e-15);
	EXPECT_NEAR(d[2], 0.0, 1e-15);

	EXPECT_NEAR(L(0, 0), 1.0, 1e-15);
	EXPECT_NEAR(L(0, 1), 0.0, 1e-15);
	EXPECT_NEAR(L(0, 2), 0.0, 1e-15);
	EXPECT_NEAR(L(1, 0), 0.666666666666666, 1e-15);
	EXPECT_NEAR(L(1, 1), 1.0, 1e-15);
	EXPECT_NEAR(L(1, 2), 0.0, 1e-15);
	EXPECT_NEAR(L(2, 0), 0.0, 1e-15);
	EXPECT_NEAR(L(2, 1), 0.0, 1e-15);
	EXPECT_NEAR(L(2, 2), 1.0, 1e-15);

	EXPECT_EQ(p[0], 2);
	EXPECT_EQ(p[1], 1);
	EXPECT_EQ(p[2], 2);
}

TEST(LinalgBackendEigen, SGMatrix_cholesky_solver)
{
	const index_t size=2;
	SGMatrix<float64_t> A(size, size);
	A(0,0)=2.0;
	A(0,1)=1.0;
	A(1,0)=1.0;
	A(1,1)=2.5;

	SGVector<float64_t> b(size);
	b[0] = 10;
	b[1] = 13;

	SGVector<float64_t> x_ref(size);
	x_ref[0] = 3;
	x_ref[1] = 4;

	SGMatrix<float64_t> L = cholesky_factor(A);
	SGVector<float64_t> x_cal = cholesky_solver(L, b);

	EXPECT_NEAR(x_ref[0], x_cal[0], 1E-15);
	EXPECT_NEAR(x_ref[1], x_cal[1], 1E-15);
	EXPECT_EQ(x_ref.size(), x_cal.size());
}

TEST(LinalgBackendEigen, SGMatrix_ldlt_solver)
{
	const index_t size = 3;
	SGMatrix<float64_t> A(size, size);
	A(0, 0) = 0.0;
	A(0, 1) = 0.0;
	A(0, 2) = 0.0;
	A(1, 0) = 0.0;
	A(1, 1) = 1.0;
	A(1, 2) = 2.0;
	A(2, 0) = 0.0;
	A(2, 1) = 2.0;
	A(2, 2) = 3.0;

	SGVector<float64_t> b(size);
	b[0] = 0.0;
	b[1] = 5.0;
	b[2] = 11.0;

	SGVector<float64_t> x_ref(size), x(size);
	x_ref[0] = 0.0;
	x_ref[1] = 7.0;
	x_ref[2] = -1.0;

	SGMatrix<float64_t> L(size, size);
	SGVector<float64_t> d(size);
	SGVector<index_t> p(size);

	linalg::ldlt_factor(A, L, d, p, true);
	x = linalg::ldlt_solver(L, d, p, b, true);
	for (auto i : range(size))
		EXPECT_NEAR(x[i], x_ref[i], 1e-15);

	linalg::ldlt_factor(A, L, d, p, false);
	x = linalg::ldlt_solver(L, d, p, b, false);
	for (auto i : range(size))
		EXPECT_NEAR(x[i], x_ref[i], 1e-15);
}
TEST(LinalgBackendEigen, SGMatrix_cross_entropy)
{
	SGMatrix<float64_t> A(4, 3);
	SGMatrix<float64_t> B(4, 3);

	int32_t size = A.num_rows * A.num_cols;
	for (float64_t i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (int32_t i = 0; i < size; i++)
		ref += A[i] * CMath::log(B[i] + 1e-30);
	ref *= -1;

	auto result = linalg::cross_entropy(A, B);
	EXPECT_NEAR(ref, result, 1e-15);
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

TEST(LinalgBackendEigen, eigensolver)
{
	const index_t n = 4;
	float64_t data[] = {0.09987322, 0.80575314, 0.79068641, 0.69989667,
	                    0.62323516, 0.16837367, 0.85027625, 0.60165948,
	                    0.04898732, 0.96701123, 0.51683275, 0.51116495,
	                    0.18277926, 0.6179262,  0.43745891, 0.63685464};
	float64_t result_eigenvectors[] = {
	    -0.63494074, 0.75831593,  -0.14014031, 0.04656076,
	    0.82257205,  -0.28671857, -0.44196422, -0.21409185,
	    -0.005932,   -0.20233724, -0.52285555, 0.82803776,
	    -0.23930111, -0.56199714, -0.57298901, -0.54642272};
	float64_t result_eigenvalues[] = {-0.6470538, -0.19125664, 0.16205101,
	                                  2.0981937};

	SGMatrix<float64_t> m(data, n, n, false);
	SGMatrix<float64_t> eigenvectors(n, n);
	SGVector<float64_t> eigenvalues(n);

	eigen_solver(m, eigenvalues, eigenvectors);

	auto args = CMath::argsort(eigenvalues);
	for (index_t i = 0; i < n; ++i)
	{
		index_t idx = args[i];
		EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-7);

		auto s =
		    CMath::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(
			    eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
			    1e-7);
	}
}

TEST(LinalgBackendEigen, eigensolver_symmetric)
{
	const index_t n = 4;
	float64_t data[] = {0.09987322, 0.80575314, 0.04898732, 0.69989667,
	                    0.80575314, 0.16837367, 0.96701123, 0.6179262,
	                    0.04898732, 0.96701123, 0.51683275, 0.43745891,
	                    0.69989667, 0.6179262,  0.43745891, 0.63685464};
	float64_t result_eigenvectors[] = {
	    -0.54618542, 0.69935447,  -0.45219663, 0.09001671,
	    -0.56171388, -0.41397154, 0.17642953,  0.69424612,
	    -0.46818396, 0.16780603,  0.73247599,  -0.46489119,
	    0.40861077,  0.55800718,  0.47735703,  0.542029037};
	float64_t result_eigenvalues[] = {-1.00663298, -0.18672196, 0.42940933,
	                                  2.18587989};

	SGMatrix<float64_t> m(data, n, n, false);
	SGMatrix<float64_t> eigenvectors(n, n);
	SGVector<float64_t> eigenvalues(n);

	eigen_solver(m, eigenvalues, eigenvectors);

	auto args = CMath::argsort(eigenvalues);
	for (index_t i = 0; i < n; ++i)
	{
		index_t idx = args[i];
		EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-7);

		auto s =
		    CMath::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(
			    eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
			    1e-7);
	}
}

TEST(LinalgBackendEigen, SGMatrix_elementwise_product)
{
	const auto m = 3;
	SGMatrix<float64_t> A(m, m);
	SGMatrix<float64_t> B(m, m);

	for (auto i : range(m * m))
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	auto result = element_prod(A, B);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(i, j) * B(i, j), 1E-15);

	result = element_prod(A, B, true, false);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(i, j), 1E-15);

	result = element_prod(A, B, false, true);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(i, j), 1E-15);

	result = element_prod(A, B, true, true);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(j, i), 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_elementwise_product_in_place)
{
	const auto m = 3;
	SGMatrix<float64_t> A(m, m);
	SGMatrix<float64_t> B(m, m);
	SGMatrix<float64_t> result(m, m);

	for (auto i : range(m * m))
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	element_prod(A, B, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(i, j) * B(i, j), 1E-15);

	element_prod(A, B, result, true, false);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(i, j), 1E-15);

	element_prod(A, B, result, false, true);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(i, j), 1E-15);

	element_prod(A, B, result, true, true);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(j, i), 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_block_elementwise_product)
{
	const index_t nrows = 2;
	const index_t ncols = 3;

	SGMatrix<float64_t> A(nrows,ncols);
	SGMatrix<float64_t> B(ncols,nrows);

	for (auto i : range(nrows))
		for (auto j : range(ncols))
		{
			A(i, j) = i * 10 + j + 1;
			B(j, i) = i + j;
		}

	const auto m = 2;
	auto A_block = linalg::block(A, 0, 0, m, m);
	auto B_block = linalg::block(B, 0, 0, m, m);
	auto result = element_prod(A_block, B_block);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(i, j) * B(i, j), 1E-15);

	result = element_prod(A_block, B_block, true, false);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(i, j), 1E-15);

	result = element_prod(A_block, B_block, false, true);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(i, j) * B(j, i), 1E-15);

	result = element_prod(A_block, B_block, true, true);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(result(i, j), A(j, i) * B(j, i), 1E-15);
}

TEST(LinalgBackendEigen, SGVector_elementwise_product)
{
	const index_t len = 4;
	SGVector<float64_t> a(len);
	SGVector<float64_t> b(len);
	SGVector<float64_t> c(len);

	for (index_t i = 0; i < len; ++i)
	{
		a[i] = i;
		b[i] = 0.5 * i;
	}

	c = element_prod(a, b);

	for (index_t i = 0; i < len; ++i)
		EXPECT_NEAR(a[i] * b[i], c[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_elementwise_product_in_place)
{
	const index_t len = 4;
	SGVector<float64_t> a(len);
	SGVector<float64_t> b(len);
	SGVector<float64_t> c(len);

	for (index_t i = 0; i < len; ++i)
	{
		a[i] = i;
		b[i] = 0.5 * i;
		c[i] = i;
	}

	element_prod(a, b, a);
	for (index_t i = 0; i < len; ++i)
		EXPECT_NEAR(c[i] * b[i], a[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_exponent)
{
	const index_t len = 4;
	SGVector<float64_t> a(len);
	a[0] = -2.4;
	a[1] = 0;
	a[2] = 0.5;
	a[3] = 3.9;
	auto result = exponent(a);

	EXPECT_NEAR(result[0], 0.090717953289413, 1E-15);
	EXPECT_NEAR(result[1], 1.0, 1E-15);
	EXPECT_NEAR(result[2], 1.648721270700128, 1E-15);
	EXPECT_NEAR(result[3], 49.40244910553017, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_exponent)
{
	const index_t n = 2;
	SGMatrix<float64_t> a(n, n);
	a[0] = -2.4;
	a[1] = 0;
	a[2] = 0.5;
	a[3] = 3.9;
	auto result = exponent(a);

	EXPECT_NEAR(result[0], 0.090717953289413, 1E-15);
	EXPECT_NEAR(result[1], 1.0, 1E-15);
	EXPECT_NEAR(result[2], 1.648721270700128, 1E-15);
	EXPECT_NEAR(result[3], 49.40244910553017, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_identity)
{
	const index_t n = 4;
	SGMatrix<float64_t> A(n, n);
	identity(A);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = 0; j < n; ++j)
			EXPECT_EQ(A.get_element(i, j), (i==j));
}

TEST(LinalgBackendEigen, logistic)
{
	SGMatrix<float64_t> A(3,3);
	SGMatrix<float64_t> B(3,3);

	range_fill(A, 0.0);
	B.zero();

	linalg::logistic(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(1.0/(1+CMath::exp(-1*A[i])), B[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_SGVector_matrix_prod)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols);
	SGVector<float64_t> b(cols);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i]=0.5 * i;
	}

	auto x = matrix_prod(A, b);

	float64_t ref[] = {10, 11.5, 13, 14.5};

	EXPECT_EQ(x.vlen, A.num_rows);
	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_SGVector_matrix_prod_transpose)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(cols, rows);
	SGVector<float64_t> b(cols);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 0.5 * i;
	}

	auto x = matrix_prod(A, b, true);

	float64_t ref[] = {7.5, 9, 10.5, 14.5};

	EXPECT_EQ(x.vlen, A.num_cols);
	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_SGVector_matrix_prod_in_place)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(rows, cols);
	SGVector<float64_t> b(cols);
	SGVector<float64_t> x(rows);

	for (index_t i = 0; i<cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 0.5 * i;
	}

	matrix_prod(A, b, x);

	float64_t ref[] = {10, 11.5, 13, 14.5};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_SGVector_matrix_prod_in_place_transpose)
{
	const index_t rows=4;
	const index_t cols=3;

	SGMatrix<float64_t> A(cols, rows);
	SGVector<float64_t> b(cols);
	SGVector<float64_t> x(rows);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 0.5 * i;
	}

	matrix_prod(A, b, x, true);

	float64_t ref[] = {7.5, 9, 10.5, 14.5};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim2, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;

	auto cal = linalg::matrix_prod(A, B);

	float64_t ref[] = {14, 17, 38, 49, 62, 81};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_transpose_A)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim2, dim1);
	SGMatrix<float64_t> B(dim2, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;

	auto cal = linalg::matrix_prod(A, B, true);

	float64_t ref[] = {7, 19, 19, 63, 31, 107};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_transpose_B)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim3, dim2);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;

	auto cal = linalg::matrix_prod(A, B, false, true);

	float64_t ref[] = {42, 51, 48, 59, 54, 67};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_transpose_A_B)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim2, dim1);
	SGMatrix<float64_t> B(dim3, dim2);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;

	auto cal = linalg::matrix_prod(A, B, true, true);

	float64_t ref[] = {21, 57, 24, 68, 27, 79};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_in_place)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim2, dim3);
	SGMatrix<float64_t> cal(dim1, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5 * i;
	cal.zero();

	linalg::matrix_prod(A, B, cal);

	float64_t ref[] = {14, 17, 38, 49, 62, 81};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_in_place_transpose_A)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim2, dim1);
	SGMatrix<float64_t> B(dim2, dim3);
	SGMatrix<float64_t> cal(dim1, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, true);

	float64_t ref[] = {7, 19, 19, 63, 31, 107};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_in_place_transpose_B)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim1, dim2);
	SGMatrix<float64_t> B(dim3, dim2);
	SGMatrix<float64_t> cal(dim1, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, false, true);

	float64_t ref[] = {42, 51, 48, 59, 54, 67};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGMatrix_matrix_product_in_place_transpose_A_B)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 3;
	SGMatrix<float64_t> A(dim2, dim1);
	SGMatrix<float64_t> B(dim3, dim2);
	SGMatrix<float64_t> cal(dim1, dim3);

	for (index_t i = 0; i < dim1*dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2*dim3; ++i)
		B[i] = 0.5*i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, true, true);

	float64_t ref[] = {21, 57, 24, 68, 27, 79};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1*dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TEST(LinalgBackendEigen, SGVector_max)
{
	SGVector<float64_t> A(9);

	float64_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 4};

	for (index_t i = 0; i < A.size(); ++i)
		A[i] = a[i];

	EXPECT_NEAR(8, max(A), 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_max)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> A(nrows, ncols);

	float64_t a[] = {1, 2, 5, 8, 3, 1, 0, -1, 12};

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

TEST(LinalgBackendEigen, SGMatrix_multiply_by_logistic_derivative)
{
	SGMatrix<float64_t> A(3, 3);
	SGMatrix<float64_t> B(3, 3);

	for (float64_t i = 0; i < 9; ++i)
	{
		A[i] = i / 9;
		B[i] = i;
	}

	linalg::multiply_by_logistic_derivative(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * A[i] * (1.0 - A[i]), B[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_multiply_by_rectified_linear_derivative)
{
	SGMatrix<float64_t> A(3, 3);
	SGMatrix<float64_t> B(3, 3);

	for (float64_t i = 0; i < 9; ++i)
	{
		A[i] = i * 0.5 - 0.5;
		B[i] = i;
	}

	multiply_by_rectified_linear_derivative(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * (A[i] != 0), B[i], 1e-15);
}

TEST(LinalgBackendEigen, SGVector_norm)
{
	const index_t n = 5;
	SGVector<float64_t> v(n);
	float64_t gt = 0;
	for (index_t i = 0; i < n; ++i)
	{
		v[i] = i;
		gt += i * i;
	}
	gt = CMath::sqrt(gt);

	auto result = norm(v);

	EXPECT_NEAR(result, gt, 1E-15);
}

TEST(LinalgBackendEigen, SGVector_qr_solver)
{
	const index_t n = 3;
	float64_t data_A[] = {0.02800922, 0.99326012, 0.15204902,
	                      0.30492837, 0.39708534, 0.40466969,
	                      0.36415317, 0.04407589, 0.9095746};
	float64_t data_b[] = {0.39461571, 0.6816856, 0.43323709};
	float64_t result[] = {0.07135206, 1.56393127, -0.23141312};

	SGMatrix<float64_t> A(data_A, n, n, false);
	SGVector<float64_t> b(data_b, n, false);

	auto x = qr_solver(A, b);

	for (index_t i = 0; i < x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1E-7);
}

TEST(LinalgBackendEigen, SGMatrix_qr_solver)
{
	const index_t n = 3, m = 2;
	float64_t data_A[] = {0.02800922, 0.99326012, 0.15204902,
	                      0.30492837, 0.39708534, 0.40466969,
	                      0.36415317, 0.04407589, 0.9095746};
	float64_t data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	float64_t result[] = {-0.73834587, 4.22750496, -1.37484721,
	                      -1.14718091, 4.49142548, -1.08282992};

	SGMatrix<float64_t> A(data_A, n, n, false);
	SGMatrix<float64_t> B(data_B, n, m, false);

	auto X = qr_solver(A, B);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1E-7);
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

TEST(LinalgBackendEigen, SGMatrix_rectified_linear)
{
	SGMatrix<float64_t> A(3, 3);
	SGMatrix<float64_t> B(3, 3);

	range_fill(A, -5.0);

	linalg::rectified_linear(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(CMath::max(0.0, A[i]), B[i], 1e-15);
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

TEST(LinalgBackendEigen, SGMatrix_softmax)
{
	SGMatrix<float64_t> A(4, 3);
	SGMatrix<float64_t> ref(4, 3);

	for (float64_t i = 0; i < 12; ++i)
		A[i] = i / 12;

	for (index_t i = 0; i < 12; ++i)
		ref[i] = CMath::exp(A[i]);

	for (index_t j = 0; j < ref.num_cols; ++j)
	{
		float64_t sum = 0;
		for (index_t i = 0; i < ref.num_rows; ++i)
			sum += ref(i, j);

		for (index_t i = 0; i < ref.num_rows; ++i)
			ref(i, j) /= sum;
	}

	linalg::softmax(A);

	for (index_t i = 0; i < 12; ++i)
		EXPECT_NEAR(ref[i], A[i], 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_squared_error)
{
	SGMatrix<float64_t> A(4, 3);
	SGMatrix<float64_t> B(4, 3);

	int32_t size = A.num_rows * A.num_cols;
	for (float64_t i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (index_t i = 0; i < size; i++)
		ref += CMath::pow(A[i] - B[i], 2);
	ref *= 0.5;

	auto result = linalg::squared_error(A, B);
	EXPECT_NEAR(ref, result, 1e-15);
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

TEST(LinalgBackendEigen, SGMatrix_symmetric_with_diag)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_NEAR(sum_symmetric(mat), 39.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_symmetric_no_diag)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_NEAR(sum_symmetric(mat, true), 36.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_symmetric_exception)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n + 1);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_THROW(sum_symmetric(mat), ShogunException);
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

TEST(LinalgBackendEigen, SGMatrix_symmetric_block_with_diag)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	float64_t sum = sum_symmetric(linalg::block(mat,1,1,2,2));
	EXPECT_NEAR(sum, 28.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_symmetric_block_no_diag)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	float64_t sum = sum_symmetric(linalg::block(mat,1,1,2,2), true);
	EXPECT_NEAR(sum, 26.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_symmetric_block_exception)
{
	const index_t n = 3;
	SGMatrix<float64_t> mat(n, n);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_THROW(sum_symmetric(linalg::block(mat,1,1,2,3)), ShogunException);
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

TEST(LinalgBackendEigen, SGMatrix_svd_jacobi_thinU)
{
	const index_t m = 5, n = 3;
	float64_t data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	float64_t result_s[] = {1.75382524, 0.56351367, 0.41124883};
	float64_t result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
	                        -0.46186125, -0.69145782, 0.29548428,  0.5718984,
	                        0.31771648,  -0.08101592, -0.27461424, 0.37170223,
	                        -0.12681555, -0.53830325, 0.69323293};

	SGMatrix<float64_t> A(data, m, n, false);
	SGMatrix<float64_t> U(m, n);
	SGVector<float64_t> s(n);

	svd(A, s, U, true, SVDAlgorithm::Jacobi);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = CMath::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-7);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-7);
}

TEST(LinalgBackendEigen, SGMatrix_svd_jacobi_fullU)
{
	const index_t m = 5, n = 3;
	float64_t data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	float64_t result_s[] = {1.75382524, 0.56351367, 0.41124883};
	float64_t result_U[] = {
	    -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
	    -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
	    -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
	    -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
	    0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

	SGMatrix<float64_t> A(data, m, n, false);
	SGMatrix<float64_t> U(m, m);
	SGVector<float64_t> s(n);

	svd(A, s, U, false, SVDAlgorithm::Jacobi);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = CMath::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-7);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-7);
}

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
TEST(LinalgBackendEigen, SGMatrix_svd_bdc_thinU)
{
	const index_t m = 5, n = 3;
	float64_t data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	float64_t result_s[] = {1.75382524, 0.56351367, 0.41124883};
	float64_t result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
	                        -0.46186125, -0.69145782, 0.29548428,  0.5718984,
	                        0.31771648,  -0.08101592, -0.27461424, 0.37170223,
	                        -0.12681555, -0.53830325, 0.69323293};

	SGMatrix<float64_t> A(data, m, n, false);
	SGMatrix<float64_t> U(m, n);
	SGVector<float64_t> s(n);

	svd(A, s, U, true, SVDAlgorithm::BidiagonalDivideConquer);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = CMath::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-7);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-7);
}

TEST(LinalgBackendEigen, SGMatrix_svd_bdc_fullU)
{
	const index_t m = 5, n = 3;
	float64_t data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	float64_t result_s[] = {1.75382524, 0.56351367, 0.41124883};
	float64_t result_U[] = {
	    -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
	    -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
	    -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
	    -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
	    0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

	SGMatrix<float64_t> A(data, m, n, false);
	SGMatrix<float64_t> U(m, m);
	SGVector<float64_t> s(n);

	svd(A, s, U, false, SVDAlgorithm::BidiagonalDivideConquer);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = CMath::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-7);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-7);
}
#endif

TEST(LinalgBackendEigen, SGMatrix_trace)
{
	const index_t n = 4;

	SGMatrix<float64_t> A(n, n);
	for (index_t i = 0; i < n*n; ++i)
		A[i] = i;

	float64_t tr = 0;
	for (index_t i = 0; i < n; ++i)
		tr += A.get_element(i, i);

	EXPECT_NEAR(trace(A), tr, 1e-15);
}

TEST(LinalgBackendEigen, SGMatrix_transpose_matrix)
{
	const index_t m = 5, n = 3;
	float64_t data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};

	SGMatrix<float64_t> A(data, m, n, false);

	auto T = transpose_matrix(A);

	for (index_t i = 0; i < m; ++i)
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(A.get_element(i, j), T.get_element(j, i), 1e-15);
}

TEST(LinalgBackendEigen, SGVector_triangular_solver_lower)
{
	const index_t n = 3;
	float64_t data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
	                      0.,          -0.27048649, -0.05919915,
	                      0.,          0.,          0.11869106};
	float64_t data_b[] = {0.39461571, 0.6816856, 0.43323709};
	float64_t result[] = {-0.42455592, -0.72571316, 0.17192745};

	SGMatrix<float64_t> L(data_L, n, n, false);
	SGVector<float64_t> b(data_b, n, false);

	auto x = triangular_solver(L, b, true);

	for (index_t i = 0; i < (index_t)x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1E-6);
}

TEST(LinalgBackendEigen, SGVector_triangular_solver_upper)
{
	const index_t n = 3;
	float64_t data_U[] = {-0.92947874, 0.,          0.,
	                      -1.1432887,  -0.27048649, 0.,
	                      -0.87119086, -0.05919915, 0.11869106};
	float64_t data_b[] = {0.39461571, 0.6816856, 0.43323709};
	float64_t result[] = {0.23681135, -3.31909306, 3.65012412};

	SGMatrix<float64_t> U(data_U, n, n, false);
	SGVector<float64_t> b(data_b, n, false);

	auto x = triangular_solver(U, b, false);

	for (index_t i = 0; i < (index_t)x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1E-6);
}

TEST(LinalgBackendEigen, SGMatrix_triangular_solver_lower)
{
	const index_t n = 3, m = 2;
	float64_t data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
	                      0.,          -0.27048649, -0.05919915,
	                      0.,          0.,          0.11869106};
	float64_t data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	float64_t result[] = {-0.82600139, 0.22050986, -3.02127745,
	                      -1.01467136, 2.08424024, -0.86262387};

	SGMatrix<float64_t> L(data_L, n, n, false);
	SGMatrix<float64_t> B(data_B, n, m, false);

	auto X = triangular_solver(L, B, true);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1E-6);
}

TEST(LinalgBackendEigen, SGMatrix_triangular_solver_upper)
{
	const index_t n = 3, m = 2;
	float64_t data_U[] = {-0.92947874, 0.,          0.,
	                      -1.1432887,  -0.27048649, 0.,
	                      -0.87119086, -0.05919915, 0.11869106};
	float64_t data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	float64_t result[] = {1.238677,    -3.91243241, 2.9315793,
	                      -2.00784647, -3.41825732, 5.54550138};

	SGMatrix<float64_t> L(data_U, n, n, false);
	SGMatrix<float64_t> B(data_B, n, m, false);

	auto X = triangular_solver(L, B, false);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1E-6);
}

TEST(LinalgBackendEigen, SGVector_zero)
{
	const index_t n = 16;
	SGVector<float64_t> a(n);
	zero(a);

	for (index_t i = 0; i < n; ++i)
		EXPECT_EQ(a[i], 0);
}

TEST(LinalgBackendEigen, SGMatrix_zero)
{
	const index_t nrows = 3, ncols = 4;
	SGMatrix<float64_t> A(nrows, ncols);
	zero(A);

	for (index_t i = 0; i < nrows*ncols; ++i)
		EXPECT_EQ(A[i], 0);
}
