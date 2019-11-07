#include "sg_gtest_utilities.h"

#include <shogun/base/range.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>

using namespace shogun;
using namespace linalg;
using namespace Eigen;

// Tolerance values for tests
template <typename T>
constexpr T get_epsilon()
{
	return std::numeric_limits<T>::epsilon() * 100;
}
template <>
constexpr floatmax_t get_epsilon()
{
	return 1e-13;
}

template <typename T>
class LinalgBackendEigenAllTypesTest : public ::testing::Test
{
};
template <typename T>
class LinalgBackendEigenNonComplexTypesTest : public ::testing::Test
{
};
template <typename T>
class LinalgBackendEigenRealTypesTest : public ::testing::Test
{
};
template <typename T>
class LinalgBackendEigenNonIntegerTypesTest : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(
    LinalgBackendEigenAllTypesTest, sg_all_primitive_types, bool, complex128_t);
SG_TYPED_TEST_CASE(LinalgBackendEigenNonComplexTypesTest, sg_non_complex_types);
SG_TYPED_TEST_CASE(LinalgBackendEigenRealTypesTest, sg_real_types);
SG_TYPED_TEST_CASE(
    LinalgBackendEigenNonIntegerTypesTest, sg_non_integer_types, complex128_t);

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;

	SGVector<TypeParam> A(9);
	SGVector<TypeParam> B(9);

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	auto result = add(A, B, alpha, beta);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(
		    alpha * A[i] + beta * B[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGMatrix<TypeParam> B(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	auto result = add(A, B, alpha, beta);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(
		    alpha * A[i] + beta * B[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_in_place)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;

	SGVector<TypeParam> A(9), B(9), C(9);

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 2 * i;
		C[i] = i;
	}

	add(A, B, A, alpha, beta);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha * C[i] + beta * B[i], A[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_in_place)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGMatrix<TypeParam> B(nrows, ncols);
	SGMatrix<TypeParam> C(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
	{
		A[i] = i;
		B[i] = 3 * i;
		C[i] = i;
	}

	add(A, B, A, alpha, beta);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(alpha * C[i] + beta * B[i], A[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_col_vec_allocated)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGVector<TypeParam> b(nrows);
	SGVector<TypeParam> result(nrows);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3 * i;

	add_col_vec(A, col, b, result, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    result[i], alpha * A.get_element(i, col) + beta * b[i],
		    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_col_vec_in_place)
{
	const TypeParam alpha = 0;
	const TypeParam beta = 3;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGVector<TypeParam> b(nrows);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 2 * i;

	add_col_vec(A, col, b, b, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    b[i], alpha * A.get_element(i, col) + beta * 2 * i,
		    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_col_vec_allocated)
{
	const TypeParam alpha = 0;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGVector<TypeParam> b(nrows);
	SGMatrix<TypeParam> result(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3 * i;

	add_col_vec(A, col, b, result, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    result.get_element(i, col),
		    alpha * A.get_element(i, col) + beta * b[i],
		    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_col_vec_in_place)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGVector<TypeParam> b(nrows);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3 * i;

	add_col_vec(A, col, b, A, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
		{
			TypeParam a = i + j * nrows;
			if (j == col)
				EXPECT_NEAR(
				    A.get_element(i, j), alpha * a + beta * b[i],
				    get_epsilon<TypeParam>());
			else
				EXPECT_EQ(A.get_element(i, j), a);
		}
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_diag)
{
	SGMatrix<TypeParam> A1(2, 3);
	SGVector<TypeParam> b1(2);

	A1(0, 0) = 1;
	A1(0, 1) = 2;
	A1(0, 2) = 3;
	A1(1, 0) = 4;
	A1(1, 1) = 5;
	A1(1, 2) = 6;

	b1[0] = 1;
	b1[1] = 2;

	const TypeParam alpha = 1.0;
	const TypeParam beta = 2.0;

	add_diag(A1, b1, alpha, beta);

	EXPECT_NEAR(A1(0, 0), 3, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(0, 1), 2, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(0, 2), 3, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 0), 4, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 1), 9, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 2), 6, get_epsilon<TypeParam>());

	// test error cases
	SGMatrix<TypeParam> A2(2, 2);
	SGVector<TypeParam> b2(3);
	SGMatrix<TypeParam> A3;
	SGVector<TypeParam> b3;
	EXPECT_THROW(add_diag(A2, b2), ShogunException);
	EXPECT_THROW(add_diag(A2, b3), ShogunException);
	EXPECT_THROW(add_diag(A3, b2), ShogunException);
	EXPECT_THROW(add_diag(A3, b3), ShogunException);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_ridge)
{
	SGMatrix<TypeParam> A1(2, 3);

	A1(0, 0) = 1;
	A1(0, 1) = 2;
	A1(0, 2) = 3;
	A1(1, 0) = 4;
	A1(1, 1) = 5;
	A1(1, 2) = 6;

	const TypeParam alpha = 1.0;

	add_ridge(A1, alpha);

	EXPECT_NEAR(A1(0, 0), 2, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(0, 1), 2, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(0, 2), 3, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 0), 4, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 1), 6, get_epsilon<TypeParam>());
	EXPECT_NEAR(A1(1, 2), 6, get_epsilon<TypeParam>());

	// test error cases
	SGMatrix<TypeParam> A2;
	EXPECT_THROW(add_ridge(A2, alpha), ShogunException);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_vector)
{
	const TypeParam alpha = 1;
	const TypeParam beta = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGMatrix<TypeParam> result(nrows, ncols);
	SGVector<TypeParam> b(nrows);

	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3 * i;
	for (index_t j = 0; j < ncols; ++j)
		for (index_t i = 0; i < nrows; ++i)
		{
			A(i, j) = i + j * ncols;
			result(i, j) = alpha * A(i, j) + beta * b[i];
		}

	add_vector(A, b, A, alpha, beta);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(A[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_scalar)
{
	const index_t n = 4;
	TypeParam s = -0.3;

	SGVector<TypeParam> a(n);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGVector<TypeParam> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_scalar)
{
	const index_t r = 4, c = 3;
	TypeParam s = 0.4;

	SGMatrix<TypeParam> a(r, c);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGMatrix<TypeParam> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_center_matrix)
{
	const index_t n = 3;
	TypeParam data[] = {0.5, 0.3, 0.4, 0.4, 0.5, 0.3, 0.3, 0.4, 0.5};
	TypeParam result[] = {0.1, -0.1, 0.0, 0.0, 0.1, -0.1, -0.1, 0.0, 0.1};

	SGMatrix<TypeParam> m(data, n, n, false);

	center_matrix(m);

	for (index_t i = 0; i < (index_t)m.size(); ++i)
		EXPECT_NEAR(m[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_llt_lower)
{
	const index_t size = 2;
	SGMatrix<TypeParam> m(size, size);
	// need to adapt the Eigen::Matrix to TypeParam type
	typedef Matrix<TypeParam, Dynamic, Dynamic> Mxx;

	m(0, 0) = 2.0;
	m(0, 1) = 1.0;
	m(1, 0) = 1.0;
	m(1, 1) = 2.5;

	// lower triangular cholesky decomposition
	SGMatrix<TypeParam> L = cholesky_factor(m);

	Map<Mxx> map_A(m.matrix, m.num_rows, m.num_cols);
	Map<Mxx> map_L(L.matrix, L.num_rows, L.num_cols);
	EXPECT_NEAR((map_A - map_L * map_L.transpose()).norm(), 0.0, 1E-6);
	EXPECT_EQ(m.num_rows, L.num_rows);
	EXPECT_EQ(m.num_cols, L.num_cols);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_llt_upper)
{
	typedef Matrix<TypeParam, Dynamic, Dynamic> Mxx;

	const index_t size = 2;
	SGMatrix<TypeParam> m(size, size);

	m(0, 0) = 2.0;
	m(0, 1) = 1.0;
	m(1, 0) = 1.0;
	m(1, 1) = 2.5;

	// upper triangular cholesky decomposition
	SGMatrix<TypeParam> U = cholesky_factor(m, false);

	Map<Mxx> map_A(m.matrix, m.num_rows, m.num_cols);
	Map<Mxx> map_U(U.matrix, U.num_rows, U.num_cols);
	EXPECT_NEAR((map_A - map_U.transpose() * map_U).norm(), 0.0, 1E-6);
	EXPECT_EQ(m.num_rows, U.num_rows);
	EXPECT_EQ(m.num_cols, U.num_cols);
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_rank_update_upper)
{
	typedef Matrix<TypeParam, Dynamic, Dynamic> Mxx;
	typedef Matrix<TypeParam, Dynamic, 1> Vx;

	const index_t size = 2;
	TypeParam alpha = 1;
	SGMatrix<TypeParam> A(size, size);
	SGMatrix<TypeParam> U(size, size);
	SGVector<TypeParam> b(size);
	Map<Mxx> A_eig(A.matrix, size, size);
	Map<Mxx> U_eig(U.matrix, size, size);
	Map<Vx> b_eig(b.vector, size);

	U(0, 0) = 2.0;
	U(0, 1) = 1.0;
	U(1, 1) = 2.5;
	b[0] = 2;
	b[1] = 3;
	A_eig = U_eig.transpose() * U_eig;

	auto A2 = A.clone();
	Map<Mxx> A2_eig(A2.matrix, A2.num_rows, A2.num_cols);
	A2(0, 0) += b[0] * b[0];
	A2(0, 1) += b[0] * b[1];
	A2(1, 0) += b[1] * b[0];
	A2(1, 1) += b[1] * b[1];

	cholesky_rank_update(U, b, alpha, false);
	EXPECT_NEAR(
	    (A2_eig - U_eig.transpose() * U_eig).norm(), 0.0,
	    get_epsilon<TypeParam>());

	cholesky_rank_update(U, b, -alpha, false);
	EXPECT_NEAR(
	    (A_eig - U_eig.transpose() * U_eig).norm(), 0.0,
	    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_rank_update_lower)
{
	typedef Matrix<TypeParam, Dynamic, Dynamic> Mxx;
	typedef Matrix<TypeParam, Dynamic, 1> Vx;

	const index_t size = 2;
	TypeParam alpha = 1;
	SGMatrix<TypeParam> A(size, size);
	SGMatrix<TypeParam> L(size, size);
	SGVector<TypeParam> b(size);
	Map<Mxx> A_eig(A.matrix, size, size);
	Map<Mxx> L_eig(L.matrix, size, size);
	Map<Vx> b_eig(b.vector, size);

	L(0, 0) = 2.0;
	L(1, 0) = 1.0;
	L(1, 1) = 2.5;
	b[0] = 2;
	b[1] = 3;
	A_eig = L_eig * L_eig.transpose();

	auto A2 = A.clone();
	Map<Mxx> A2_eig(A2.matrix, A2.num_rows, A2.num_cols);
	A2(0, 0) += b[0] * b[0];
	A2(0, 1) += b[0] * b[1];
	A2(1, 0) += b[1] * b[0];
	A2(1, 1) += b[1] * b[1];

	cholesky_rank_update(L, b, alpha);
	EXPECT_NEAR(
	    (A2_eig - L_eig * L_eig.transpose()).norm(), 0.0,
	    get_epsilon<TypeParam>());

	cholesky_rank_update(L, b, -alpha);
	EXPECT_NEAR(
	    (A_eig - L_eig * L_eig.transpose()).norm(), 0.0,
	    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_ldlt_lower)
{
	const index_t size = 3;
	SGMatrix<TypeParam> m(size, size);
	m(0, 0) = 0.0;
	m(0, 1) = 0.0;
	m(0, 2) = 0.0;
	m(1, 0) = 0.0;
	m(1, 1) = 1.0;
	m(1, 2) = 2.0;
	m(2, 0) = 0.0;
	m(2, 1) = 2.0;
	m(2, 2) = 3.0;

	SGMatrix<TypeParam> L(size, size);
	SGVector<TypeParam> d(size);
	SGVector<index_t> p(size);

	linalg::ldlt_factor(m, L, d, p);

	EXPECT_NEAR(d[0], 3.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(d[1], -0.333333333333333, get_epsilon<TypeParam>());
	EXPECT_NEAR(d[2], 0.0, get_epsilon<TypeParam>());

	EXPECT_NEAR(L(0, 0), 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(0, 1), 0.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(0, 2), 0.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(1, 0), 0.666666666666666, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(1, 1), 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(1, 2), 0.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(2, 0), 0.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(2, 1), 0.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(L(2, 2), 1.0, get_epsilon<TypeParam>());

	EXPECT_EQ(p[0], 2);
	EXPECT_EQ(p[1], 1);
	EXPECT_EQ(p[2], 2);
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_solver)
{
	const index_t size = 2;
	SGMatrix<TypeParam> A(size, size);
	A(0, 0) = 2.0;
	A(0, 1) = 1.0;
	A(1, 0) = 1.0;
	A(1, 1) = 2.5;

	SGVector<TypeParam> b(size);
	b[0] = 10;
	b[1] = 13;

	SGVector<TypeParam> x_ref(size);
	x_ref[0] = 3;
	x_ref[1] = 4;

	SGMatrix<TypeParam> L = cholesky_factor(A);
	SGVector<TypeParam> x_cal = cholesky_solver(L, b);

	EXPECT_NEAR(x_ref[0], x_cal[0], get_epsilon<TypeParam>());
	EXPECT_NEAR(x_ref[1], x_cal[1], get_epsilon<TypeParam>());
	EXPECT_EQ(x_ref.size(), x_cal.size());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_ldlt_solver)
{
	const index_t size = 3;
	SGMatrix<TypeParam> A(size, size);
	A(0, 0) = 0.0;
	A(0, 1) = 0.0;
	A(0, 2) = 0.0;
	A(1, 0) = 0.0;
	A(1, 1) = 1.0;
	A(1, 2) = 2.0;
	A(2, 0) = 0.0;
	A(2, 1) = 2.0;
	A(2, 2) = 3.0;

	SGVector<TypeParam> b(size);
	b[0] = 0.0;
	b[1] = 5.0;
	b[2] = 11.0;

	SGVector<TypeParam> x_ref(size), x(size);
	x_ref[0] = 0.0;
	x_ref[1] = 7.0;
	x_ref[2] = -1.0;

	SGMatrix<TypeParam> L(size, size);
	SGVector<TypeParam> d(size);
	SGVector<index_t> p(size);

	linalg::ldlt_factor(A, L, d, p, true);
	x = linalg::ldlt_solver(L, d, p, b, true);
	for (auto i : range(size))
		EXPECT_NEAR(x[i], x_ref[i], get_epsilon<TypeParam>());

	linalg::ldlt_factor(A, L, d, p, false);
	x = linalg::ldlt_solver(L, d, p, b, false);
	for (auto i : range(size))
		EXPECT_NEAR(x[i], x_ref[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cross_entropy)
{
	SGMatrix<TypeParam> A(4, 3);
	SGMatrix<TypeParam> B(4, 3);

	uint32_t size = A.num_rows * A.num_cols;
	for (TypeParam i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (uint32_t i = 0; i < size; i++)
		ref += A[i] * std::log(B[i] + 1e-15);
	ref *= -1;

	auto result = linalg::cross_entropy(A, B);
	EXPECT_NEAR(ref, result, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_psd)
{
	TypeParam A_data[] = {2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
	// inverse generated by scipy pinv
	TypeParam scipy_result_data[] = {0.75, 0.5,  0.25, 0.5, 1.0,
	                                 0.5,  0.25, 0.5,  0.75};

	SGMatrix<TypeParam> A(A_data, 3, 3, false);
	SGMatrix<TypeParam> result(scipy_result_data, 3, 3, false);

	SGMatrix<TypeParam> identity_matrix(3, 3);
	linalg::identity(identity_matrix);
	// using symmetric eigen solver
	SGMatrix<TypeParam> A_pinvh(3, 3);
	linalg::pinvh(A, A_pinvh);
	// using singular value decomposition
	SGMatrix<TypeParam> A_pinv(3, 3);
	linalg::pinv(A, A_pinv);
	SGMatrix<TypeParam> I_check = linalg::matrix_prod(A, A_pinvh);
	for (auto i : range(3))
	{
		for (auto j : range(3))
		{
			EXPECT_NEAR(
			    identity_matrix(i, j), I_check(i, j), get_epsilon<TypeParam>());
			EXPECT_NEAR(result(i, j), A_pinvh(i, j), get_epsilon<TypeParam>());
			EXPECT_NEAR(result(i, j), A_pinv(i, j), get_epsilon<TypeParam>());
		}
	}
	// no memory errors
	EXPECT_NO_THROW(linalg::pinvh(A, A));
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_2x4)
{
	SGMatrix<TypeParam> A(2, 4);
	A(0, 0) = 1;
	A(0, 1) = 1;
	A(0, 2) = 1;
	A(0, 3) = 1;
	A(1, 0) = 5;
	A(1, 1) = 7;
	A(1, 2) = 7;
	A(1, 3) = 9;

	SGMatrix<TypeParam> identity_matrix(2, 2);
	linalg::identity(identity_matrix);
	SGMatrix<TypeParam> A_pinverse(4, 2);
	linalg::pinv(A, A_pinverse);
	SGMatrix<TypeParam> I_check = linalg::matrix_prod(A, A_pinverse);
	for (auto i : range(2))
	{
		for (auto j : range(2))
		{
			EXPECT_NEAR(
			    identity_matrix(i, j), I_check(i, j), get_epsilon<TypeParam>());
		}
	}

	// compare result with scipy pinv
	EXPECT_NEAR(A_pinverse(0, 0), 2.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(0, 1), -0.25, get_epsilon<TypeParam>());

	EXPECT_NEAR(A_pinverse(1, 0), 0.25, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(1, 1), 0.0, get_epsilon<TypeParam>());

	EXPECT_NEAR(A_pinverse(2, 0), 0.25, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(2, 1), 0.0, get_epsilon<TypeParam>());

	EXPECT_NEAR(A_pinverse(3, 0), -1.5, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(3, 1), 0.25, get_epsilon<TypeParam>());

	// incorrect dimension
	EXPECT_THROW(linalg::pinv(A, A), ShogunException);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_4x2)
{
	SGMatrix<TypeParam> A(4, 2);
	A(0, 0) = 2.0;
	A(0, 1) = -0.25;
	A(1, 0) = 0.25;
	A(1, 1) = 0.0;
	A(2, 0) = 0.25;
	A(2, 1) = 0.0;
	A(3, 0) = -1.5;
	A(3, 1) = 0.25;

	SGMatrix<TypeParam> identity_matrix(2, 2);
	linalg::identity(identity_matrix);
	SGMatrix<TypeParam> A_pinverse(2, 4);
	linalg::pinv(A, A_pinverse);
	SGMatrix<TypeParam> I_check = linalg::matrix_prod(A_pinverse, A);
	for (auto i : range(2))
	{
		for (auto j : range(2))
		{
			EXPECT_NEAR(
			    identity_matrix(i, j), I_check(i, j), get_epsilon<TypeParam>());
		}
	}
	// compare with results from scipy
	EXPECT_NEAR(A_pinverse(0, 0), 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(0, 1), 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(0, 2), 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(0, 3), 1.0, get_epsilon<TypeParam>());

	EXPECT_NEAR(A_pinverse(1, 0), 5.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(1, 1), 7.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(1, 2), 7.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(A_pinverse(1, 3), 9.0, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_dot)
{
	const index_t size = 3;
	SGVector<TypeParam> a(size), b(size);
	a.range_fill(0);
	b.range_fill(0);

	auto result = dot(a, b);

	EXPECT_NEAR(result, 5, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, eigensolver)
{
	const index_t n = 4;
	TypeParam data[] = {0.09987322, 0.80575314, 0.79068641, 0.69989667,
	                    0.62323516, 0.16837367, 0.85027625, 0.60165948,
	                    0.04898732, 0.96701123, 0.51683275, 0.51116495,
	                    0.18277926, 0.6179262,  0.43745891, 0.63685464};
	TypeParam result_eigenvectors[] = {
	    -0.63494074, 0.75831593,   -0.1401403109, 0.04656076,
	    0.82257205,  -0.286718557, -0.44196422,   -0.214091861,
	    -0.005932,   -0.20233723,  -0.52285555,   0.82803776,
	    -0.23930111, -0.56199714,  -0.57298901,   -0.54642272};
	TypeParam result_eigenvalues[] = {-0.6470538, -0.19125664, 0.16205101,
	                                  2.0981937};

	SGMatrix<TypeParam> m(data, n, n, false);
	SGMatrix<TypeParam> eigenvectors(n, n);
	SGVector<TypeParam> eigenvalues(n);

	eigen_solver(m, eigenvalues, eigenvectors);

	auto args = Math::argsort(eigenvalues);
	for (index_t i = 0; i < n; ++i)
	{
		index_t idx = args[i];
		EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-6);

		auto s =
		    Math::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(
			    eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
			    1e-6);
	}
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, eigensolver_symmetric)
{
	const index_t n = 4;
	TypeParam data[] = {0.09987322, 0.80575314, 0.04898732, 0.69989667,
	                    0.80575314, 0.16837367, 0.96701123, 0.6179262,
	                    0.04898732, 0.96701123, 0.51683275, 0.43745891,
	                    0.69989667, 0.6179262,  0.43745891, 0.63685464};
	TypeParam result_eigenvectors[] = {
	    -0.54618542, 0.69935447,  -0.45219663, 0.09001671,
	    -0.56171388, -0.41397154, 0.17642953,  0.69424612,
	    -0.46818396, 0.16780603,  0.73247599,  -0.46489119,
	    0.40861077,  0.55800718,  0.47735703,  0.542029037};
	TypeParam result_eigenvalues[] = {-1.00663298, -0.18672196, 0.42940933,
	                                  2.18587989};

	SGMatrix<TypeParam> m(data, n, n, false);
	SGMatrix<TypeParam> eigenvectors(n, n);
	SGVector<TypeParam> eigenvalues(n);

	eigen_solver(m, eigenvalues, eigenvectors);

	auto args = Math::argsort(eigenvalues);
	for (index_t i = 0; i < n; ++i)
	{
		index_t idx = args[i];
		EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-5);

		auto s =
		    Math::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(
			    eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
			    1e-5);
	}
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_elementwise_product)
{
	const auto m = 2;
	SGMatrix<TypeParam> A(m, m);
	SGMatrix<TypeParam> B(m, m);

	for (auto i : range(m * m))
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	auto result = element_prod(A, B);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(i, j) * B(i, j), get_epsilon<TypeParam>());

	result = element_prod(A, B, true, false);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(i, j), get_epsilon<TypeParam>());

	result = element_prod(A, B, false, true);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(i, j), get_epsilon<TypeParam>());

	result = element_prod(A, B, true, true);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(j, i), get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest, SGMatrix_elementwise_product_in_place)
{
	const auto m = 2;
	SGMatrix<TypeParam> A(m, m);
	SGMatrix<TypeParam> B(m, m);
	SGMatrix<TypeParam> result(m, m);

	for (auto i : range(m * m))
	{
		A[i] = i;
		B[i] = 2 * i;
	}

	element_prod(A, B, result);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(i, j) * B(i, j), get_epsilon<TypeParam>());

	element_prod(A, B, result, true, false);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(i, j), get_epsilon<TypeParam>());

	element_prod(A, B, result, false, true);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(i, j), get_epsilon<TypeParam>());

	element_prod(A, B, result, true, true);
	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(j, i), get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_elementwise_product)
{
	const index_t nrows = 2;
	const index_t ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);
	SGMatrix<TypeParam> B(ncols, nrows);

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
			EXPECT_NEAR(
			    result(i, j), A(i, j) * B(i, j), get_epsilon<TypeParam>());

	result = element_prod(A_block, B_block, true, false);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(i, j), get_epsilon<TypeParam>());

	result = element_prod(A_block, B_block, false, true);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(i, j) * B(j, i), get_epsilon<TypeParam>());

	result = element_prod(A_block, B_block, true, true);

	ASSERT_EQ(result.num_rows, m);
	ASSERT_EQ(result.num_cols, m);

	for (auto i : range(m))
		for (auto j : range(m))
			EXPECT_NEAR(
			    result(i, j), A(j, i) * B(j, i), get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_elementwise_product)
{
	const index_t len = 4;
	SGVector<TypeParam> a(len);
	SGVector<TypeParam> b(len);
	SGVector<TypeParam> c(len);

	for (index_t i = 0; i < len; ++i)
	{
		a[i] = i;
		b[i] = 2 * i;
	}

	c = element_prod(a, b);

	for (index_t i = 0; i < len; ++i)
		EXPECT_NEAR(a[i] * b[i], c[i], get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest, SGVector_elementwise_product_in_place)
{
	const index_t len = 4;
	SGVector<TypeParam> a(len);
	SGVector<TypeParam> b(len);
	SGVector<TypeParam> c(len);

	for (index_t i = 0; i < len; ++i)
	{
		a[i] = i;
		b[i] = 2 * i;
		c[i] = i;
	}

	element_prod(a, b, a);
	for (index_t i = 0; i < len; ++i)
		EXPECT_NEAR(c[i] * b[i], a[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_exponent)
{
	const index_t len = 4;
	SGVector<TypeParam> a(len);
	a[0] = 0;
	a[1] = 1;
	a[2] = 2;
	a[3] = 3;
	auto result = exponent(a);

	EXPECT_NEAR(result[0], 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[1], 2.718281828459045, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[2], 7.3890560989306495, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[3], 20.085536923187664, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_exponent)
{
	const index_t n = 2;
	SGMatrix<TypeParam> a(n, n);
	a[0] = 0;
	a[1] = 1;
	a[2] = 2;
	a[3] = 3;
	auto result = exponent(a);

	EXPECT_NEAR(result[0], 1.0, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[1], 2.718281828459045, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[2], 7.3890560989306495, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[3], 20.085536923187664, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_identity)
{
	const index_t n = 4;
	SGMatrix<TypeParam> A(n, n);
	identity(A);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = 0; j < n; ++j)
			EXPECT_EQ(A.get_element(i, j), (i == j));
}

// TODO: write test for int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, logistic)
{
	SGMatrix<TypeParam> A(3, 3);
	SGMatrix<TypeParam> B(3, 3);

	for (index_t i = 0; i < 9; ++i)
		A[i] = i;
	B.zero();

	linalg::logistic(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(
		    1.0 / (1 + std::exp(-1 * A[i])), B[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_SGVector_matrix_prod)
{
	const index_t rows = 4;
	const index_t cols = 3;

	SGMatrix<TypeParam> A(rows, cols);
	SGVector<TypeParam> b(cols);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 2 * i;
	}

	auto x = matrix_prod(A, b);

	TypeParam ref[] = {40, 46, 52, 58};

	EXPECT_EQ(x.vlen, A.num_rows);
	for (index_t i = 0; i < rows; ++i)
		EXPECT_NEAR(x[i], ref[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_matrix_prod_transpose)
{
	const index_t rows = 4;
	const index_t cols = 3;

	SGMatrix<TypeParam> A(cols, rows);
	SGVector<TypeParam> b(cols);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 2 * i;
	}

	auto x = matrix_prod(A, b, true);

	TypeParam ref[] = {30, 36, 42};

	EXPECT_EQ(x.vlen, A.num_cols);
	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest, SGMatrix_SGVector_matrix_prod_in_place)
{
	const index_t rows = 4;
	const index_t cols = 3;

	SGMatrix<TypeParam> A(rows, cols);
	SGVector<TypeParam> b(cols);
	SGVector<TypeParam> x(rows);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(j, i) = i * rows + j;
		b[i] = 2 * i;
	}

	matrix_prod(A, b, x);

	TypeParam ref[] = {40, 46, 52, 58};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest,
    SGMatrix_SGVector_matrix_prod_in_place_transpose)
{
	const index_t rows = 4;
	const index_t cols = 3;

	SGMatrix<TypeParam> A(cols, rows);
	SGVector<TypeParam> b(cols);
	SGVector<TypeParam> x(rows);

	for (index_t i = 0; i < cols; ++i)
	{
		for (index_t j = 0; j < rows; ++j)
			A(i, j) = i * cols + j;
		b[i] = 2 * i;
	}

	matrix_prod(A, b, x, true);

	TypeParam ref[] = {30, 36, 42};

	for (index_t i = 0; i < cols; ++i)
		EXPECT_NEAR(x[i], ref[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product)
{
	const index_t dim1 = 2, dim2 = 4, dim3 = 2;
	SGMatrix<TypeParam> A(dim1, dim2);
	SGMatrix<TypeParam> B(dim2, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;

	auto cal = linalg::matrix_prod(A, B);

	TypeParam ref[] = {28, 34, 76, 98};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_A)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim2, dim1);
	SGMatrix<TypeParam> B(dim2, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;

	auto cal = linalg::matrix_prod(A, B, true);

	TypeParam ref[] = {5, 14, 14, 50, 23, 86};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_B)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim1, dim2);
	SGMatrix<TypeParam> B(dim3, dim2);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;

	auto cal = linalg::matrix_prod(A, B, false, true);

	TypeParam ref[] = {30, 39, 36, 48, 42, 57};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_A_B)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim2, dim1);
	SGMatrix<TypeParam> B(dim3, dim2);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;

	auto cal = linalg::matrix_prod(A, B, true, true);

	TypeParam ref[] = {15, 42, 18, 54, 21, 66};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_in_place)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim1, dim2);
	SGMatrix<TypeParam> B(dim2, dim3);
	SGMatrix<TypeParam> cal(dim1, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;
	cal.zero();

	linalg::matrix_prod(A, B, cal);

	TypeParam ref[] = {10, 13, 28, 40, 46, 67};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest,
    SGMatrix_matrix_product_in_place_transpose_A)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim2, dim1);
	SGMatrix<TypeParam> B(dim2, dim3);
	SGMatrix<TypeParam> cal(dim1, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, true);

	TypeParam ref[] = {5, 14, 14, 50, 23, 86};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest,
    SGMatrix_matrix_product_in_place_transpose_B)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim1, dim2);
	SGMatrix<TypeParam> B(dim3, dim2);
	SGMatrix<TypeParam> cal(dim1, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, false, true);

	TypeParam ref[] = {30, 39, 36, 48, 42, 57};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest,
    SGMatrix_matrix_product_in_place_transpose_A_B)
{
	const index_t dim1 = 2, dim2 = 3, dim3 = 3;
	SGMatrix<TypeParam> A(dim2, dim1);
	SGMatrix<TypeParam> B(dim3, dim2);
	SGMatrix<TypeParam> cal(dim1, dim3);

	for (index_t i = 0; i < dim1 * dim2; ++i)
		A[i] = i;
	for (index_t i = 0; i < dim2 * dim3; ++i)
		B[i] = i;
	cal.zero();

	linalg::matrix_prod(A, B, cal, true, true);

	TypeParam ref[] = {15, 42, 18, 54, 21, 66};

	EXPECT_EQ(dim1, cal.num_rows);
	EXPECT_EQ(dim3, cal.num_cols);
	for (index_t i = 0; i < dim1 * dim3; ++i)
		EXPECT_EQ(ref[i], cal[i]);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_max)
{
	SGVector<TypeParam> A(9);

	TypeParam a[] = {1, 2, 5, 8, 3, 1, 0, 2, 4};

	for (index_t i = 0; i < A.size(); ++i)
		A[i] = a[i];

	EXPECT_NEAR(8, max(A), get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_max)
{
	const index_t nrows = 3, ncols = 3;
	SGMatrix<TypeParam> A(nrows, ncols);

	TypeParam a[] = {1, 2, 5, 8, 3, 1, 0, 2, 12};

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = a[i];

	EXPECT_NEAR(12, max(A), get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_mean)
{
	const index_t size = 9;
	SGVector<TypeParam> vec(size);
	vec.range_fill(0);

	auto result = mean(vec);

	EXPECT_NEAR(result, 4, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_mean)
{
	const index_t nrows = 3, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = mean(mat);

	EXPECT_NEAR(result, 4, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, Scalar_update_mean)
{
	const index_t n = 1e5;
	const TypeParam scale = 2;

	TypeParam mean_increasing = 0;
	TypeParam mean_decreasing = 0;
	for (index_t i = 0; i < n; i++)
	{
		TypeParam datum_inc(scale * i);
		TypeParam datum_dec(scale * (n - 1 - i));
		update_mean(mean_increasing, datum_inc, i + 1);
		update_mean(mean_decreasing, datum_dec, i + 1);
	}
	TypeParam mean_true((scale * (n - 1)) / 2.0);

	auto epsilon = n * get_epsilon<TypeParam>();
	EXPECT_NEAR(mean_increasing, mean_true, epsilon);
	EXPECT_NEAR(mean_decreasing, mean_true, epsilon);
	EXPECT_THROW(
	    update_mean(mean_increasing, mean_increasing, 0), ShogunException);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_update_mean)
{
	const index_t n = 1e5;
	const index_t length = 3;
	const TypeParam scale = 2;

	SGVector<TypeParam> mean_increasing(length);
	SGVector<TypeParam> mean_decreasing(length);
	SGVector<TypeParam>::fill_vector(mean_increasing, length, 0);
	SGVector<TypeParam>::fill_vector(mean_decreasing, length, 0);

	SGVector<TypeParam> datum(length);
	for (int i = 0; i < n; i++)
	{
		datum.range_fill(i);
		datum.scale(scale);
		update_mean(mean_increasing, datum, i + 1);
		datum.range_fill(n - 1 - i);
		datum.scale(scale);
		update_mean(mean_decreasing, datum, i + 1);
	}

	auto epsilon = n * get_epsilon<TypeParam>();
	for (int j = 0; j < length; j++)
	{
		EXPECT_NEAR(
		    mean_increasing[j], ((2 * j + n - 1) * scale) / 2.0, epsilon);
		EXPECT_NEAR(
		    mean_decreasing[j], ((2 * j + n - 1) * scale) / 2.0, epsilon);
	}
	EXPECT_THROW(
	    update_mean(mean_increasing, mean_increasing, 0), ShogunException);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_std_deviation_colwise)
{
	const index_t nrows = 3, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = std_deviation(mat);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    result.get_element(i), 2.449489742783178, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_std_deviation)
{
	const index_t nrows = 3, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);
	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = std_deviation(mat, false);

	EXPECT_NEAR(
	    result.get_element(0), 2.581988897471611, get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenAllTypesTest, SGMatrix_multiply_by_logistic_derivative)
{
	SGMatrix<TypeParam> A(3, 3);
	SGMatrix<TypeParam> B(3, 3);

	for (TypeParam i = 9; i < 9; i += 9)
	{
		A[i] = i / 9;
		B[i] = i;
	}

	linalg::multiply_by_logistic_derivative(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * A[i] * (1.0 - A[i]), B[i], get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenNonIntegerTypesTest,
    SGMatrix_multiply_by_rectified_linear_derivative)
{
	SGMatrix<TypeParam> A(3, 3);
	SGMatrix<TypeParam> B(3, 3);

	for (TypeParam i = 0; i < 9; ++i)
	{
		A[i] = i * 0.5 - 0.5;
		B[i] = i;
	}

	multiply_by_rectified_linear_derivative(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(i * (A[i] != 0), B[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_norm)
{
	const index_t n = 24;
	SGVector<TypeParam> v(n);
	TypeParam gt = 0;
	for (index_t i = 0; i < n; ++i)
	{
		v[i] = i;
		gt += i * i;
	}

	gt = std::sqrt(gt);

	auto result = norm(v);

	EXPECT_NEAR(result, gt, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_qr_solver)
{
	const index_t n = 3;
	TypeParam data_A[] = {0.02800922, 0.99326012, 0.15204902,
	                      0.30492837, 0.39708534, 0.40466969,
	                      0.36415317, 0.04407589, 0.9095746};
	TypeParam data_b[] = {0.39461571, 0.6816856, 0.43323709};
	TypeParam result[] = {0.07135206, 1.56393127, -0.23141312};

	SGMatrix<TypeParam> A(data_A, n, n, false);
	SGVector<TypeParam> b(data_b, n, false);

	auto x = qr_solver(A, b);

	for (index_t i = 0; i < x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1e-6);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_qr_solver)
{
	const index_t n = 3, m = 2;
	TypeParam data_A[] = {0.02800922, 0.99326012, 0.15204902,
	                      0.30492837, 0.39708534, 0.40466969,
	                      0.36415317, 0.04407589, 0.9095746};
	TypeParam data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	TypeParam result[] = {-0.73834587, 4.22750496, -1.37484721,
	                      -1.14718091, 4.49142548, -1.08282992};

	SGMatrix<TypeParam> A(data_A, n, n, false);
	SGMatrix<TypeParam> B(data_B, n, m, false);

	auto X = qr_solver(A, B);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1e-5);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_range_fill)
{
	const index_t size = 5;
	SGVector<TypeParam> vec(size);
	TypeParam start = 1;
	range_fill(vec, start);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(vec[i], i + 1, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_range_fill)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);
	TypeParam start = 1;
	range_fill(mat, start);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(mat[i], i + 1, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rectified_linear)
{
	SGMatrix<TypeParam> A(3, 3);
	SGMatrix<TypeParam> B(3, 3);
	TypeParam start = 1;
	range_fill(A, start);

	linalg::rectified_linear(A, B);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(
		    Math::max(static_cast<TypeParam>(0.0), A[i]), B[i],
		    get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_scale)
{
	const index_t size = 5;
	const TypeParam alpha = 2;
	SGVector<TypeParam> a(size);
	a.range_fill(0);

	auto result = scale(a, alpha);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * a[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_scale)
{
	const TypeParam alpha = 2;
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> A(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;

	auto result = scale(A, alpha);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(alpha * A[i], result[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_scale_in_place)
{
	const index_t size = 5;
	const TypeParam alpha = 2;
	SGVector<TypeParam> a(size);
	a.range_fill(0);

	scale(a, a, alpha);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(alpha * i, a[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_scale_in_place)
{
	const TypeParam alpha = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		A[i] = i;

	scale(A, A, alpha);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(alpha * i, A[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_scale_cols_in_place)
{
	const TypeParam alpha = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<TypeParam> A(nrows, ncols);

	for(index_t r = 0; r < nrows; ++r)
		for(index_t c = 0; c < ncols; ++c)
			A(r, c) = c * nrows + r;

	SGVector<TypeParam> alphas(ncols);
	alphas.range_fill(1);

	scale(A, A, alphas);

	for(index_t r = 0; r < nrows; ++r)
		for(index_t c = 0; c < ncols; ++c)
			EXPECT_EQ(A(r, c), alphas[c] * (c * nrows + r));
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_set_const)
{
	const index_t size = 5;
	const TypeParam value = 2;
	SGVector<TypeParam> a(size);

	set_const(a, value);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], value, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_set_const)
{
	const index_t nrows = 2, ncols = 3;
	const TypeParam value = 2;
	SGMatrix<TypeParam> a(nrows, ncols);

	set_const(a, value);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_NEAR(a[i], value, get_epsilon<TypeParam>());
}

// TODO: extend to all types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_softmax)
{
	SGMatrix<TypeParam> A(4, 3);
	SGMatrix<TypeParam> ref(4, 3);

	for (TypeParam i = 0; i < 12; ++i)
		A[i] = i / 12;

	for (index_t i = 0; i < 12; ++i)
		ref[i] = std::exp(A[i]);

	for (index_t j = 0; j < ref.num_cols; ++j)
	{
		TypeParam sum = 0;
		for (index_t i = 0; i < ref.num_rows; ++i)
			sum += ref(i, j);

		for (index_t i = 0; i < ref.num_rows; ++i)
			ref(i, j) /= sum;
	}

	linalg::softmax(A);

	for (index_t i = 0; i < 12; ++i)
		EXPECT_NEAR(ref[i], A[i], get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_sum)
{
	const index_t size = 10;
	SGVector<TypeParam> vec(size);
	vec.range_fill(0);

	auto result = sum(vec);

	EXPECT_NEAR(result, 45, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = sum(mat);

	EXPECT_NEAR(result, 15, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	auto result = sum(mat, true);

	EXPECT_NEAR(result, 12, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_with_diag)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n);
	mat.set_const(1);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_NEAR(sum_symmetric(mat), 39, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_no_diag)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n);
	mat.set_const(1);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_NEAR(sum_symmetric(mat, true), 36, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_exception)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n + 1);
	mat.set_const(1.0);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	EXPECT_THROW(sum_symmetric(mat), ShogunException);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_sum)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = 0; j < n; ++j)
			mat(i, j) = i * 10 + j + 1;

	auto result = sum(linalg::block(mat, 0, 0, 2, 3));
	EXPECT_NEAR(result, 42.0, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_with_diag)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n);
	mat.set_const(1);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	TypeParam sum = sum_symmetric(linalg::block(mat, 1, 1, 2, 2));
	EXPECT_NEAR(sum, 28, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_no_diag)
{
	const index_t n = 3;
	SGMatrix<TypeParam> mat(n, n);
	mat.set_const(1);

	for (index_t i = 0; i < n; ++i)
		for (index_t j = i + 1; j < n; ++j)
		{
			mat(i, j) = i * 10 + j + 1;
			mat(j, i) = mat(i, j);
		}

	TypeParam sum = sum_symmetric(linalg::block(mat, 1, 1, 2, 2), true);
	EXPECT_NEAR(sum, 26, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_exception)
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

	EXPECT_THROW(
	    sum_symmetric(linalg::block(mat, 1, 1, 2, 3)), ShogunException);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_colwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<TypeParam> result = colwise_sum(mat);

	for (index_t j = 0; j < ncols; ++j)
	{
		TypeParam sum = 0;
		for (index_t i = 0; i < nrows; ++i)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[j], get_epsilon<TypeParam>());
	}
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_colwise_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<TypeParam> result = colwise_sum(mat, true);

	EXPECT_NEAR(result[0], 1, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[1], 2, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[2], 9, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_colwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> mat(nrows, ncols);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
			mat(i, j) = i * 10 + j + 1;

	auto result = colwise_sum(linalg::block(mat, 0, 0, 2, 3));

	for (index_t j = 0; j < ncols; ++j)
	{
		TypeParam sum = 0;
		for (index_t i = 0; i < nrows; ++i)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[j], get_epsilon<TypeParam>());
	}
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rowwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<TypeParam> result = rowwise_sum(mat);

	for (index_t i = 0; i < nrows; ++i)
	{
		TypeParam sum = 0;
		for (index_t j = 0; j < ncols; ++j)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[i], get_epsilon<TypeParam>());
	}
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rowwise_sum_no_diag)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows * ncols; ++i)
		mat[i] = i;

	SGVector<TypeParam> result = rowwise_sum(mat, true);

	EXPECT_NEAR(result[0], 6, get_epsilon<TypeParam>());
	EXPECT_NEAR(result[1], 6, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_rowwise_sum)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<TypeParam> mat(nrows, ncols);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
			mat(i, j) = i * 10 + j + 1;

	auto result = rowwise_sum(linalg::block(mat, 0, 0, 2, 3));

	for (index_t i = 0; i < nrows; ++i)
	{
		TypeParam sum = 0;
		for (index_t j = 0; j < ncols; ++j)
			sum += mat(i, j);
		EXPECT_NEAR(sum, result[i], get_epsilon<TypeParam>());
	}
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_jacobi_thinU)
{
	const index_t m = 5, n = 3;
	TypeParam data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	TypeParam result_s[] = {1.75382524, 0.56351367, 0.41124883};
	TypeParam result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
	                        -0.46186125, -0.69145782, 0.29548428,  0.5718984,
	                        0.31771648,  -0.08101592, -0.27461424, 0.37170223,
	                        -0.12681555, -0.53830325, 0.69323293};

	SGMatrix<TypeParam> A(data, m, n, false);
	SGMatrix<TypeParam> U(m, n);
	SGVector<TypeParam> s(n);

	svd(A, s, U, true, SVDAlgorithm::Jacobi);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = Math::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_jacobi_fullU)
{
	const index_t m = 5, n = 3;
	TypeParam data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	TypeParam result_s[] = {1.75382524, 0.56351367, 0.41124883};
	TypeParam result_U[] = {
	    -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
	    -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
	    -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
	    -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
	    0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

	SGMatrix<TypeParam> A(data, m, n, false);
	SGMatrix<TypeParam> U(m, m);
	SGVector<TypeParam> s(n);

	svd(A, s, U, false, SVDAlgorithm::Jacobi);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = Math::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_bdc_thinU)
{
	const index_t m = 5, n = 3;
	TypeParam data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	TypeParam result_s[] = {1.75382524, 0.56351367, 0.41124883};
	TypeParam result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
	                        -0.46186125, -0.69145782, 0.29548428,  0.5718984,
	                        0.31771648,  -0.08101592, -0.27461424, 0.37170223,
	                        -0.12681555, -0.53830325, 0.69323293};

	SGMatrix<TypeParam> A(data, m, n, false);
	SGMatrix<TypeParam> U(m, n);
	SGVector<TypeParam> s(n);

	svd(A, s, U, true, SVDAlgorithm::BidiagonalDivideConquer);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = Math::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_bdc_fullU)
{
	const index_t m = 5, n = 3;
	TypeParam data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
	                    0.30786772, 0.25503552, 0.34367041, 0.66491478,
	                    0.20488809, 0.5734351,  0.87179189, 0.07139643,
	                    0.28540373, 0.06264684, 0.56204061};
	TypeParam result_s[] = {1.75382524, 0.56351367, 0.41124883};
	TypeParam result_U[] = {
	    -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
	    -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
	    -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
	    -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
	    0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

	SGMatrix<TypeParam> A(data, m, n, false);
	SGMatrix<TypeParam> U(m, m);
	SGVector<TypeParam> s(n);

	svd(A, s, U, false, SVDAlgorithm::BidiagonalDivideConquer);

	for (index_t i = 0; i < n; ++i)
	{
		auto c = Math::sign(U[i * m] * result_U[i * m]);
		for (index_t j = 0; j < m; ++j)
			EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
	}
	for (index_t i = 0; i < (index_t)s.size(); ++i)
		EXPECT_NEAR(s[i], result_s[i], 1e-6);
}
#endif

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_trace)
{
	const index_t n = 4;

	SGMatrix<TypeParam> A(n, n);
	for (index_t i = 0; i < n * n; ++i)
		A[i] = i;

	TypeParam tr = 0;
	for (index_t i = 0; i < n; ++i)
		tr += A.get_element(i, i);

	EXPECT_NEAR(trace(A), tr, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_trace_dot)
{
	const index_t n = 2;
	SGMatrix<TypeParam> A(n, n), B(n, n);
	for (index_t i = 0; i < n * n; ++i)
	{
		A[i] = i;
		B[i] = i * 2;
	}

	auto C = matrix_prod(A, B);
	auto tr = 0.0;
	for (auto i : range(n))
		tr += C(i, i);

	EXPECT_NEAR(tr, trace_dot(A, B), get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_transpose_matrix)
{
	const index_t m = 5, n = 3;
	SGMatrix<TypeParam> A(m, n);
	linalg::range_fill(A, static_cast<TypeParam>(1));

	auto T = transpose_matrix(A);

	for (index_t i = 0; i < m; ++i)
		for (index_t j = 0; j < n; ++j)
			EXPECT_NEAR(
			    A.get_element(i, j), T.get_element(j, i),
			    get_epsilon<TypeParam>());
}

TYPED_TEST(
    LinalgBackendEigenNonIntegerTypesTest, SGVector_triangular_solver_lower)
{
	const index_t n = 3;
	TypeParam data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
	                      0.,          -0.27048649, -0.05919915,
	                      0.,          0.,          0.11869106};
	TypeParam data_b[] = {0.39461571, 0.6816856, 0.43323709};
	TypeParam result[] = {-0.42455592, -0.72571316, 0.17192745};

	SGMatrix<TypeParam> L(data_L, n, n, false);
	SGVector<TypeParam> b(data_b, n, false);

	auto x = triangular_solver(L, b, true);

	for (index_t i = 0; i < (index_t)x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1e-6);
}

TYPED_TEST(
    LinalgBackendEigenNonIntegerTypesTest, SGVector_triangular_solver_upper)
{
	const index_t n = 3;
	TypeParam data_U[] = {-0.92947874, 0.,          0.,
	                      -1.1432887,  -0.27048649, 0.,
	                      -0.87119086, -0.05919915, 0.11869106};
	TypeParam data_b[] = {0.39461571, 0.6816856, 0.43323709};
	TypeParam result[] = {0.23681135, -3.31909306, 3.65012412};

	SGMatrix<TypeParam> U(data_U, n, n, false);
	SGVector<TypeParam> b(data_b, n, false);

	auto x = triangular_solver(U, b, false);

	for (index_t i = 0; i < (index_t)x.size(); ++i)
		EXPECT_NEAR(x[i], result[i], 1e-6);
}

TYPED_TEST(
    LinalgBackendEigenNonIntegerTypesTest, SGMatrix_triangular_solver_lower)
{
	const index_t n = 3, m = 2;
	TypeParam data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
	                      0.,          -0.27048649, -0.05919915,
	                      0.,          0.,          0.11869106};
	TypeParam data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	TypeParam result[] = {-0.82600139, 0.22050986, -3.02127745,
	                      -1.01467136, 2.08424024, -0.86262387};

	SGMatrix<TypeParam> L(data_L, n, n, false);
	SGMatrix<TypeParam> B(data_B, n, m, false);

	auto X = triangular_solver(L, B, true);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1e-6);
}

TYPED_TEST(
    LinalgBackendEigenNonIntegerTypesTest, SGMatrix_triangular_solver_upper)
{
	const index_t n = 3, m = 2;
	TypeParam data_U[] = {-0.92947874, 0.,          0.,
	                      -1.1432887,  -0.27048649, 0.,
	                      -0.87119086, -0.05919915, 0.11869106};
	TypeParam data_B[] = {0.76775073, 0.88471312, 0.34795225,
	                      0.94311546, 0.59630347, 0.65820143};
	TypeParam result[] = {1.238677,    -3.91243241, 2.9315793,
	                      -2.00784647, -3.41825732, 5.54550138};

	SGMatrix<TypeParam> L(data_U, n, n, false);
	SGMatrix<TypeParam> B(data_B, n, m, false);

	auto X = triangular_solver(L, B, false);

	for (index_t i = 0; i < (index_t)X.size(); ++i)
		EXPECT_NEAR(X[i], result[i], 1e-6);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_zero)
{
	const index_t n = 16;
	SGVector<TypeParam> a(n);
	zero(a);

	for (index_t i = 0; i < n; ++i)
		EXPECT_EQ(a[i], 0);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_zero)
{
	const index_t nrows = 3, ncols = 4;
	SGMatrix<TypeParam> A(nrows, ncols);
	zero(A);

	for (index_t i = 0; i < nrows * ncols; ++i)
		EXPECT_EQ(A[i], 0);
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rank_update)
{
	typedef Matrix<TypeParam, Dynamic, Dynamic> Mxx;
	typedef Matrix<TypeParam, 1, Dynamic> Vxd;

	const index_t size = 2;
	SGMatrix<TypeParam> A(size, size);
	SGVector<TypeParam> b(size);
	Map<Mxx> A_eig(A.matrix, size, size);
	Map<Vxd> b_eig(b.vector, size);

	A(0, 0) = 2.0;
	A(1, 0) = 1.0;
	A(0, 1) = 1.0;
	A(1, 1) = 2.5;
	b[0] = 2;
	b[1] = 3;

	auto A2 = A.clone();
	Map<Mxx> A2_eig(A2.matrix, size, size);
	A2(0, 0) += b[0] * b[0];
	A2(0, 1) += b[0] * b[1];
	A2(1, 0) += b[1] * b[0];
	A2(1, 1) += b[1] * b[1];

	rank_update(A, b, static_cast<TypeParam>(1));
	EXPECT_NEAR((A2_eig - A_eig).norm(), 0, get_epsilon<TypeParam>());

	rank_update(A, b, static_cast<TypeParam>(-1));
	EXPECT_NEAR((A_eig - A_eig).norm(), 0, get_epsilon<TypeParam>());
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_squared_error)
{
	SGMatrix<TypeParam> A(4, 3);
	SGMatrix<TypeParam> B(4, 3);

	int32_t size = A.num_rows * A.num_cols;
	for (float64_t i = 0; i < size; ++i)
	{
		A[i] = i / size;
		B[i] = (i / size) * 0.5;
	}

	float64_t ref = 0;
	for (index_t i = 0; i < size; i++)
		ref += std::pow(A[i] - B[i], 2);
	ref *= 0.5;

	auto result = linalg::squared_error(A, B);
	EXPECT_NEAR(ref, result, get_epsilon<TypeParam>());
}
