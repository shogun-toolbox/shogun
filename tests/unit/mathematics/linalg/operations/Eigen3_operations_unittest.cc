#include <gtest/gtest.h>

#include <shogun/base/range.h>
#include <shogun/lib/config.h>
#include <shogun/lib/exception/ShogunException.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>

using namespace shogun;
using namespace linalg;
using namespace Eigen;

template <typename T>
class LinalgBackendEigenAllTypesTest: public ::testing::Test { };
template <typename T>
class LinalgBackendEigenNonComplexTypesTest: public ::testing::Test { };
template <typename T>
class LinalgBackendEigenRealTypesTest: public ::testing::Test { };
template <typename T>
class LinalgBackendEigenNonIntegerTypesTest: public ::testing::Test { };

// TODO: make global definitions
// Definition of the 4 groups of Shogun types (shogun/mathematics/linalg/LinalgBackendBase.h)
// TODO: add bool, chars and complex128_t types
typedef ::testing::Types<int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float32_t, float64_t, floatmax_t> AllTypes;
typedef ::testing::Types<int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float32_t, float64_t, floatmax_t> NonComplexTypes;
typedef ::testing::Types<float32_t, float64_t, floatmax_t> RealTypes;
// TODO: add complex128_t type
typedef ::testing::Types<float32_t, float64_t, floatmax_t> NonIntegerTypes;

TYPED_TEST_CASE(LinalgBackendEigenAllTypesTest, AllTypes);
TYPED_TEST_CASE(LinalgBackendEigenNonComplexTypesTest, NonComplexTypes);
TYPED_TEST_CASE(LinalgBackendEigenRealTypesTest, RealTypes);
TYPED_TEST_CASE(LinalgBackendEigenNonIntegerTypesTest, NonIntegerTypes);


template <typename ST>
void check_SGVector_add() {

    const ST alpha = 1;
    const ST beta = 2;

    SGVector<ST> A(9);
    SGVector<ST> B(9);

    for (index_t i = 0; i < 9; ++i)
    {
        A[i] = i;
        B[i] = 2*i;
    }

    auto result = add(A, B, alpha, beta);

    for (index_t i = 0; i < 9; ++i)
        EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_add() {

    const ST alpha = 1;
    const ST beta = 2;
    const index_t nrows = 2, ncols = 3;

    SGMatrix<ST> A(nrows, ncols);
    SGMatrix<ST> B(nrows, ncols);

    for (index_t i = 0; i < nrows*ncols; ++i)
    {
        A[i] = i;
        B[i] = 2*i;
    }

    auto result = add(A, B, alpha, beta);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

template <typename ST>
void check_SGVector_add_in_place() {
    const ST alpha = 1;
    const ST beta = 2;

    SGVector<ST> A(9), B(9), C(9);

    for (index_t i = 0; i < 9; ++i)
    {
    A[i] = i;
    B[i] = 2*i;
    C[i] = i;
    }

    add(A, B, A, alpha, beta);

    for (index_t i = 0; i < 9; ++i)
    EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_add_in_place() {

    const ST alpha = 1;
    const ST beta = 2;
    const index_t nrows = 2, ncols = 3;

    SGMatrix<ST> A(nrows, ncols);
    SGMatrix<ST> B(nrows, ncols);
    SGMatrix<ST> C(nrows, ncols);

    for (index_t i = 0; i < nrows*ncols; ++i)
    {
        A[i] = i;
        B[i] = 3*i;
        C[i] = i;
    }

    add(A, B, A, alpha, beta);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(alpha*C[i]+beta*B[i], A[i], 1e-15);
}

template <typename ST>
void check_SGVector_add_col_vec_allocated() {

    const ST alpha = 1;
    const ST beta = 2;
    const index_t nrows = 2, ncols = 3;
    const index_t col = 1;

    SGMatrix<ST> A(nrows, ncols);
    SGVector<ST> b(nrows);
    SGVector<ST> result(nrows);

    for (index_t i = 0; i < nrows * ncols; ++i)
        A[i] = i;
    for (index_t i = 0; i < nrows; ++i)
        b[i] = 3 * i;

    add_col_vec(A, col, b, result, alpha, beta);

    for (index_t i = 0; i < nrows; ++i)
        EXPECT_NEAR(result[i], alpha * A.get_element(i, col) + beta * b[i], 1e-15);
}

template <typename ST>
void check_SGVector_add_col_vec_in_place()
{
    const ST alpha = 0;
    const ST beta = 3;
    const index_t nrows = 2, ncols = 3;
    const index_t col = 1;

    SGMatrix<ST> A(nrows, ncols);
    SGVector<ST> b(nrows);

    for (index_t i = 0; i < nrows*ncols; ++i)
        A[i] = i;
    for (index_t i = 0; i < nrows; ++i)
        b[i] = 2*i;

    add_col_vec(A, col, b, b, alpha, beta);

    for (index_t i = 0; i < nrows; ++i)
        EXPECT_NEAR(b[i], alpha*A.get_element(i, col)+beta*2*i, 1e-15);
}

template <typename ST>
void check_SGMatrix_add_col_vec_allocated()
{
	const ST alpha = 0;
	const ST beta = 2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<ST> A(nrows, ncols);
	SGVector<ST> b(nrows);
	SGMatrix<ST> result(nrows, ncols);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3*i;

	add_col_vec(A, col, b, result, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		EXPECT_NEAR(
		    result.get_element(i, col),
		    alpha * A.get_element(i, col) + beta * b[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_add_col_vec_in_place()
{
	const ST alpha = 1;
	const ST beta = 2;
	const index_t nrows = 2, ncols = 3;
	const index_t col = 1;

	SGMatrix<ST> A(nrows, ncols);
	SGVector<ST> b(nrows);

	for (index_t i = 0; i < nrows*ncols; ++i)
		A[i] = i;
	for (index_t i = 0; i < nrows; ++i)
		b[i] = 3*i;

	add_col_vec(A, col, b, A, alpha, beta);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
		{
			ST a = i+j*nrows;
			if (j == col)
				EXPECT_NEAR(A.get_element(i, j), alpha*a+beta*b[i], 1e-15);
			else
				EXPECT_EQ(A.get_element(i,j), a);
		}
}

template <typename ST>
void check_add_diag()
{
	SGMatrix<ST> A1(2, 3);
	SGVector<ST> b1(2);

	A1(0, 0) = 1;
	A1(0, 1) = 2;
	A1(0, 2) = 3;
	A1(1, 0) = 4;
	A1(1, 1) = 5;
	A1(1, 2) = 6;

	b1[0] = 1;
	b1[1] = 2;

	const ST alpha = 1.0;
	const ST beta = 2.0;

	add_diag(A1, b1, alpha, beta);

	EXPECT_NEAR(A1(0, 0), 3, 1e-15);
	EXPECT_NEAR(A1(0, 1), 2, 1e-15);
	EXPECT_NEAR(A1(0, 2), 3, 1e-15);
	EXPECT_NEAR(A1(1, 0), 4, 1e-15);
	EXPECT_NEAR(A1(1, 1), 9, 1e-15);
	EXPECT_NEAR(A1(1, 2), 6, 1e-15);

	// test error cases
	SGMatrix<ST> A2(2, 2);
	SGVector<ST> b2(3);
	SGMatrix<ST> A3;
	SGVector<ST> b3;
	EXPECT_THROW(add_diag(A2, b2), ShogunException);
	EXPECT_THROW(add_diag(A2, b3), ShogunException);
	EXPECT_THROW(add_diag(A3, b2), ShogunException);
	EXPECT_THROW(add_diag(A3, b3), ShogunException);
}

template <typename ST>
void check_add_ridge()
{
	SGMatrix<ST> A1(2, 3);

	A1(0, 0) = 1;
	A1(0, 1) = 2;
	A1(0, 2) = 3;
	A1(1, 0) = 4;
	A1(1, 1) = 5;
	A1(1, 2) = 6;

	const ST alpha = 1.0;

	add_ridge(A1, alpha);

	EXPECT_NEAR(A1(0, 0), 2, 1e-15);
	EXPECT_NEAR(A1(0, 1), 2, 1e-15);
	EXPECT_NEAR(A1(0, 2), 3, 1e-15);
	EXPECT_NEAR(A1(1, 0), 4, 1e-15);
	EXPECT_NEAR(A1(1, 1), 6, 1e-15);
	EXPECT_NEAR(A1(1, 2), 6, 1e-15);

	// test error cases
	SGMatrix<ST> A2;
	EXPECT_THROW(add_ridge(A2, alpha), ShogunException);
}

template <typename ST>
void check_add_vector()
{
	const ST alpha = 1;
	const ST beta = 2;
	const index_t nrows = 2, ncols = 3;

	SGMatrix<ST> A(nrows, ncols);
	SGMatrix<ST> result(nrows, ncols);
	SGVector<ST> b(nrows);

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
		EXPECT_NEAR(A[i], result[i], 1e-15);
}

template <typename ST>
void check_SGVector_add_scalar()
{
	const index_t n = 4;
	ST s = -0.3;

	SGVector<ST> a(n);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGVector<ST> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, 1e-15);
}

template <typename ST>
void check_SGMatrix_add_scalar()
{
	const index_t r = 4, c = 3;
	ST s = 0.4;

	SGMatrix<ST> a(r, c);
	for (index_t i = 0; i < (index_t)a.size(); ++i)
		a[i] = i;
	SGMatrix<ST> orig = a.clone();

	add_scalar(a, s);

	for (index_t i = 0; i < (index_t)a.size(); ++i)
		EXPECT_NEAR(a[i], orig[i] + s, 1e-15);
}

template <typename ST>
void check_SGMatrix_center_matrix()
{
	const index_t n = 3;
	ST data[] = {0.5, 0.3, 0.4,
	             0.4, 0.5, 0.3,
	             0.3, 0.4, 0.5};
	ST result[] = {0.1, -0.1, 0.0,
	               0.0, 0.1, -0.1,
	               -0.1, 0.0, 0.1};

	SGMatrix<ST> m(data, n, n, false);

	center_matrix(m);

	for (index_t i = 0; i < (index_t)m.size(); ++i)
		EXPECT_NEAR(m[i], result[i], 1e-7);
}

template <typename ST>
void check_SGMatrix_cholesky_llt_lower()
{
	const index_t size=2;
	SGMatrix<ST> m(size, size);
	typedef Matrix<ST, Dynamic, Dynamic> Mxx; // need to adapt the Eigen::Matrix to ST type

	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	//lower triangular cholesky decomposition
	SGMatrix<ST> L = cholesky_factor(m);


	Map<Mxx> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<Mxx> map_L(L.matrix,L.num_rows,L.num_cols);
	EXPECT_NEAR((map_A-map_L*map_L.transpose()).norm(),
		0.0, 1E-6); // need to change the treshold to work with floats
	EXPECT_EQ(m.num_rows, L.num_rows);
	EXPECT_EQ(m.num_cols, L.num_cols);
}

template <typename ST>
void check_SGMatrix_cholesky_llt_upper()
{
    typedef Matrix<ST, Dynamic, Dynamic> Mxx;

    const index_t size=2;
    SGMatrix<ST> m(size, size);

    m(0,0)=2.0;
    m(0,1)=1.0;
    m(1,0)=1.0;
    m(1,1)=2.5;

    //upper triangular cholesky decomposition
    SGMatrix<ST> U = cholesky_factor(m,false);

    Map<Mxx> map_A(m.matrix,m.num_rows,m.num_cols);
    Map<Mxx> map_U(U.matrix,U.num_rows,U.num_cols);
    EXPECT_NEAR((map_A-map_U.transpose()*map_U).norm(),
    0.0, 1E-6); // need to change the treshold to work with floats
    EXPECT_EQ(m.num_rows, U.num_rows);
    EXPECT_EQ(m.num_cols, U.num_cols);
}

template <typename ST>
void check_SGMatrix_cholesky_rank_update_upper()
{
    typedef Matrix<ST, Dynamic, Dynamic> Mxx;
    typedef Matrix<ST, Dynamic, 1> Vx;

    const index_t size = 2;
    ST alpha = 1;
    SGMatrix<ST> A(size, size);
    SGMatrix<ST> U(size, size);
    SGVector<ST> b(size);
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
    EXPECT_NEAR((A2_eig - U_eig.transpose() * U_eig).norm(), 0.0, 1e-5);  // need to change the treshold to work with floats

    cholesky_rank_update(U, b, -alpha, false);
    EXPECT_NEAR((A_eig - U_eig.transpose() * U_eig).norm(), 0.0, 1e-5);  // need to change the treshold to work with floats
}

template <typename ST>
void check_SGMatrix_cholesky_rank_update_lower()
{
    typedef Matrix<ST, Dynamic, Dynamic> Mxx;
    typedef Matrix<ST, Dynamic, 1> Vx;

    const index_t size = 2;
    ST alpha = 1;
    SGMatrix<ST> A(size, size);
    SGMatrix<ST> L(size, size);
    SGVector<ST> b(size);
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
    EXPECT_NEAR((A2_eig - L_eig * L_eig.transpose()).norm(), 0.0, 1e-5);

    cholesky_rank_update(L, b, -alpha);
    EXPECT_NEAR((A_eig - L_eig * L_eig.transpose()).norm(), 0.0, 1e-5);
}

template <typename ST>
void check_SGMatrix_cholesky_ldlt_lower()
{
    const index_t size = 3;
    SGMatrix<ST> m(size, size);
    m(0, 0) = 0.0;
    m(0, 1) = 0.0;
    m(0, 2) = 0.0;
    m(1, 0) = 0.0;
    m(1, 1) = 1.0;
    m(1, 2) = 2.0;
    m(2, 0) = 0.0;
    m(2, 1) = 2.0;
    m(2, 2) = 3.0;

    SGMatrix<ST> L(size, size);
    SGVector<ST> d(size);
    SGVector<index_t> p(size);

    linalg::ldlt_factor(m, L, d, p);

    EXPECT_NEAR(d[0], 3.0, 1e-15);
    EXPECT_NEAR(d[1], -0.333333333333333, 1e-7);
    EXPECT_NEAR(d[2], 0.0, 1e-15);

    EXPECT_NEAR(L(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(L(0, 1), 0.0, 1e-15);
    EXPECT_NEAR(L(0, 2), 0.0, 1e-15);
    EXPECT_NEAR(L(1, 0), 0.666666666666666, 1e-7);
    EXPECT_NEAR(L(1, 1), 1.0, 1e-15);
    EXPECT_NEAR(L(1, 2), 0.0, 1e-15);
    EXPECT_NEAR(L(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(L(2, 1), 0.0, 1e-15);
    EXPECT_NEAR(L(2, 2), 1.0, 1e-15);

    EXPECT_EQ(p[0], 2);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 2);
}

template <typename ST>
void check_SGMatrix_cholesky_solver()
{
    const index_t size=2;
    SGMatrix<ST> A(size, size);
    A(0,0)=2.0;
    A(0,1)=1.0;
    A(1,0)=1.0;
    A(1,1)=2.5;

    SGVector<ST> b(size);
    b[0] = 10;
    b[1] = 13;

    SGVector<ST> x_ref(size);
    x_ref[0] = 3;
    x_ref[1] = 4;

    SGMatrix<ST> L = cholesky_factor(A);
    SGVector<ST> x_cal = cholesky_solver(L, b);

    EXPECT_NEAR(x_ref[0], x_cal[0], 1E-15);
    EXPECT_NEAR(x_ref[1], x_cal[1], 1E-15);
    EXPECT_EQ(x_ref.size(), x_cal.size());
}

template <typename ST>
void check_SGMatrix_ldlt_solver()
{
    const index_t size = 3;
    SGMatrix<ST> A(size, size);
    A(0, 0) = 0.0;
    A(0, 1) = 0.0;
    A(0, 2) = 0.0;
    A(1, 0) = 0.0;
    A(1, 1) = 1.0;
    A(1, 2) = 2.0;
    A(2, 0) = 0.0;
    A(2, 1) = 2.0;
    A(2, 2) = 3.0;

    SGVector<ST> b(size);
    b[0] = 0.0;
    b[1] = 5.0;
    b[2] = 11.0;

    SGVector<ST> x_ref(size), x(size);
    x_ref[0] = 0.0;
    x_ref[1] = 7.0;
    x_ref[2] = -1.0;

    SGMatrix<ST> L(size, size);
    SGVector<ST> d(size);
    SGVector<index_t> p(size);

    linalg::ldlt_factor(A, L, d, p, true);
    x = linalg::ldlt_solver(L, d, p, b, true);
    for (auto i : range(size))
        EXPECT_NEAR(x[i], x_ref[i], 1e-6);

    linalg::ldlt_factor(A, L, d, p, false);
    x = linalg::ldlt_solver(L, d, p, b, false);
    for (auto i : range(size))
        EXPECT_NEAR(x[i], x_ref[i], 1e-6);
}

template <typename ST>
void check_SGMatrix_cross_entropy()
{
    SGMatrix<ST> A(4, 3);
    SGMatrix<ST> B(4, 3);

    uint32_t size = A.num_rows * A.num_cols;
    for (ST i = 0; i < size; ++i)
    {
        A[i] = i / size;
        B[i] = (i / size) * 0.5;
    }

    float64_t ref = 0;
    for (uint32_t i = 0; i < size; i++)
        ref += A[i] * std::log(B[i] + 1e-30);
    ref *= -1;

    auto result = linalg::cross_entropy(A, B);
    EXPECT_NEAR(ref, result, 1e-6);
}

template <typename ST>
void check_SGMatrix_pinv_psd()
{
    ST A_data[] = {2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
    // inverse generated by scipy pinv
    ST scipy_result_data[] = {0.75, 0.5,  0.25, 0.5, 1.0,
                                     0.5,  0.25, 0.5,  0.75};

    SGMatrix<ST> A(A_data, 3, 3, false);
    SGMatrix<ST> result(scipy_result_data, 3, 3, false);

    SGMatrix<ST> identity_matrix(3, 3);
    linalg::identity(identity_matrix);
    // using symmetric eigen solver
    SGMatrix<ST> A_pinvh(3, 3);
    linalg::pinvh(A, A_pinvh);
    // using singular value decomposition
    SGMatrix<ST> A_pinv(3, 3);
    linalg::pinv(A, A_pinv);
    SGMatrix<ST> I_check = linalg::matrix_prod(A, A_pinvh);
    for (auto i : range(3))
    {
        for (auto j : range(3))
        {
            EXPECT_NEAR(identity_matrix(i, j), I_check(i, j), 1e-6);
            EXPECT_NEAR(result(i, j), A_pinvh(i, j), 1e-6);
            EXPECT_NEAR(result(i, j), A_pinv(i, j), 1e-6);
        }
    }
    // no memory errors
    EXPECT_NO_THROW(linalg::pinvh(A, A));
}

template <typename ST>
void check_SGMatrix_pinv_2x4()
{
    SGMatrix<ST> A(2, 4);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(0, 2) = 1;
    A(0, 3) = 1;
    A(1, 0) = 5;
    A(1, 1) = 7;
    A(1, 2) = 7;
    A(1, 3) = 9;

    SGMatrix<ST> identity_matrix(2, 2);
    linalg::identity(identity_matrix);
    SGMatrix<ST> A_pinverse(4, 2);
    linalg::pinv(A, A_pinverse);
    SGMatrix<ST> I_check = linalg::matrix_prod(A, A_pinverse);
    for (auto i : range(2))
    {
        for (auto j : range(2))
        {
            EXPECT_NEAR(identity_matrix(i, j), I_check(i, j), 1e-6);
        }
    }

    // compare result with scipy pinv
    EXPECT_NEAR(A_pinverse(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(A_pinverse(0, 1), -0.25, 1e-6);

    EXPECT_NEAR(A_pinverse(1, 0), 0.25, 1e-6);
    EXPECT_NEAR(A_pinverse(1, 1), 0.0, 1e-6);

    EXPECT_NEAR(A_pinverse(2, 0), 0.25, 1e-6);
    EXPECT_NEAR(A_pinverse(2, 1), 0.0, 1e-6);

    EXPECT_NEAR(A_pinverse(3, 0), -1.5, 1e-6);
    EXPECT_NEAR(A_pinverse(3, 1), 0.25, 1e-6);

    // incorrect dimension
    EXPECT_THROW(linalg::pinv(A, A), ShogunException);
}

template <typename ST>
void check_SGMatrix_pinv_4x2()
{
    SGMatrix<ST> A(4, 2);
    A(0, 0) = 2.0;
    A(0, 1) = -0.25;
    A(1, 0) = 0.25;
    A(1, 1) = 0.0;
    A(2, 0) = 0.25;
    A(2, 1) = 0.0;
    A(3, 0) = -1.5;
    A(3, 1) = 0.25;

    SGMatrix<ST> identity_matrix(2, 2);
    linalg::identity(identity_matrix);
    SGMatrix<ST> A_pinverse(2, 4);
    linalg::pinv(A, A_pinverse);
    SGMatrix<ST> I_check = linalg::matrix_prod(A_pinverse, A);
    for (auto i : range(2))
    {
        for (auto j : range(2))
        {
        EXPECT_NEAR(identity_matrix(i, j), I_check(i, j), 1e-05);
        }
    }
    // compare with results from scipy
    EXPECT_NEAR(A_pinverse(0, 0), 1.0, 1e-05);
    EXPECT_NEAR(A_pinverse(0, 1), 1.0, 1e-05);
    EXPECT_NEAR(A_pinverse(0, 2), 1.0, 1e-05);
    EXPECT_NEAR(A_pinverse(0, 3), 1.0, 1e-05);

    EXPECT_NEAR(A_pinverse(1, 0), 5.0, 1e-05);
    EXPECT_NEAR(A_pinverse(1, 1), 7.0, 1e-05);
    EXPECT_NEAR(A_pinverse(1, 2), 7.0, 1e-05);
    EXPECT_NEAR(A_pinverse(1, 3), 9.0, 1e-05);
}

template <typename ST>
void check_SGVector_dot()
{
    const index_t size = 3;
    SGVector<ST> a(size), b(size);
    a.range_fill(0);
    b.range_fill(0);

    auto result = dot(a, b);

    EXPECT_NEAR(result, 5, 1E-15);
}

template <typename ST>
void check_eigensolver()
{
    const index_t n = 4;
    ST data[] = {0.09987322, 0.80575314, 0.79068641, 0.69989667,
                        0.62323516, 0.16837367, 0.85027625, 0.60165948,
                        0.04898732, 0.96701123, 0.51683275, 0.51116495,
                        0.18277926, 0.6179262,  0.43745891, 0.63685464};
    ST result_eigenvectors[] = {
            -0.63494074, 0.75831593,  -0.14014031, 0.04656076,
            0.82257205,  -0.28671857, -0.44196422, -0.21409185,
            -0.005932,   -0.20233724, -0.52285555, 0.82803776,
            -0.23930111, -0.56199714, -0.57298901, -0.54642272};
    ST result_eigenvalues[] = {-0.6470538, -0.19125664, 0.16205101,
                                      2.0981937};

    SGMatrix<ST> m(data, n, n, false);
    SGMatrix<ST> eigenvectors(n, n);
    SGVector<ST> eigenvalues(n);

    eigen_solver(m, eigenvalues, eigenvectors);

    auto args = CMath::argsort(eigenvalues);
    for (index_t i = 0; i < n; ++i)
    {
    index_t idx = args[i];
    EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-6);

    auto s =
            CMath::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
    for (index_t j = 0; j < n; ++j)
        EXPECT_NEAR(
                eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
        1e-6);
    }
}

template <typename ST>
void check_eigensolver_symmetric()
{
    const index_t n = 4;
    ST data[] = {0.09987322, 0.80575314, 0.04898732, 0.69989667,
                        0.80575314, 0.16837367, 0.96701123, 0.6179262,
                        0.04898732, 0.96701123, 0.51683275, 0.43745891,
                        0.69989667, 0.6179262,  0.43745891, 0.63685464};
    ST result_eigenvectors[] = {
            -0.54618542, 0.69935447,  -0.45219663, 0.09001671,
            -0.56171388, -0.41397154, 0.17642953,  0.69424612,
            -0.46818396, 0.16780603,  0.73247599,  -0.46489119,
            0.40861077,  0.55800718,  0.47735703,  0.542029037};
    ST result_eigenvalues[] = {-1.00663298, -0.18672196, 0.42940933,
                                      2.18587989};

    SGMatrix<ST> m(data, n, n, false);
    SGMatrix<ST> eigenvectors(n, n);
    SGVector<ST> eigenvalues(n);

    eigen_solver(m, eigenvalues, eigenvectors);

    auto args = CMath::argsort(eigenvalues);
    for (index_t i = 0; i < n; ++i)
    {
        index_t idx = args[i];
        EXPECT_NEAR(eigenvalues[idx], result_eigenvalues[i], 1e-5);

        auto s =
                CMath::sign(eigenvectors[idx * n] * result_eigenvectors[i * n]);
        for (index_t j = 0; j < n; ++j)
            EXPECT_NEAR(
                    eigenvectors[idx * n + j], s * result_eigenvectors[i * n + j],
            1e-5);
    }
}

template <typename ST>
void check_SGMatrix_elementwise_product()
{
    const auto m = 2;
    SGMatrix<ST> A(m, m);
    SGMatrix<ST> B(m, m);

    for (auto i : range(m * m))
    {
        A[i] = i;
        B[i] = 2*i;
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

template <typename ST>
void check_SGMatrix_elementwise_product_in_place()
{
    const auto m = 2;
    SGMatrix<ST> A(m, m);
    SGMatrix<ST> B(m, m);
    SGMatrix<ST> result(m, m);

    for (auto i : range(m * m))
    {
        A[i] = i;
        B[i] = 2*i;
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

template <typename ST>
void check_SGMatrix_block_elementwise_product()
{
    const index_t nrows = 2;
    const index_t ncols = 3;

    SGMatrix<ST> A(nrows,ncols);
    SGMatrix<ST> B(ncols,nrows);

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

template <typename ST>
void check_SGVector_elementwise_product()
{
    const index_t len = 4;
    SGVector<ST> a(len);
    SGVector<ST> b(len);
    SGVector<ST> c(len);

    for (index_t i = 0; i < len; ++i)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    c = element_prod(a, b);

    for (index_t i = 0; i < len; ++i)
        EXPECT_NEAR(a[i] * b[i], c[i], 1e-15);
}

template <typename ST>
void check_SGVector_elementwise_product_in_place()
{
    const index_t len = 4;
    SGVector<ST> a(len);
    SGVector<ST> b(len);
    SGVector<ST> c(len);

    for (index_t i = 0; i < len; ++i)
    {
        a[i] = i;
        b[i] = 2 * i;
        c[i] = i;
    }

    element_prod(a, b, a);
    for (index_t i = 0; i < len; ++i)
        EXPECT_NEAR(c[i] * b[i], a[i], 1e-15);
}

template <typename ST>
void check_SGVector_exponent()
{
    const index_t len = 4;
    SGVector<ST> a(len);
    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;
    auto result = exponent(a);

    EXPECT_NEAR(result[0], 1.0, 1E-15);
    EXPECT_NEAR(result[1], 2.718281828459045, 1E-6);
    EXPECT_NEAR(result[2], 7.3890560989306495, 1E-6);
    EXPECT_NEAR(result[3], 20.085536923187664, 1E-6);
}

template <typename ST>
void check_SGMatrix_exponent()
{
    const index_t n = 2;
    SGMatrix<ST> a(n, n);
    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;
    auto result = exponent(a);

    EXPECT_NEAR(result[0], 1.0, 1E-15);
    EXPECT_NEAR(result[1], 2.718281828459045, 1E-6);
    EXPECT_NEAR(result[2], 7.3890560989306495, 1E-6);
    EXPECT_NEAR(result[3], 20.085536923187664, 1E-6);
}

template <typename ST>
void check_SGMatrix_identity()
{
    const index_t n = 4;
    SGMatrix<ST> A(n, n);
    identity(A);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < n; ++j)
            EXPECT_EQ(A.get_element(i, j), (i==j));
}

template <typename ST>
void check_logistic()
{
    SGMatrix<ST> A(3,3);
    SGMatrix<ST> B(3,3);

    for (index_t i = 0; i < 9; ++i)
        A[i] = i;
    B.zero();

    linalg::logistic(A, B);

    for (index_t i = 0; i < 9; ++i)
        EXPECT_NEAR(1.0 / (1 + std::exp(-1 * A[i])), B[i], 1e-7);
}

template <typename ST>
void check_SGMatrix_SGVector_matrix_prod()
{
    const index_t rows=4;
    const index_t cols=3;

    SGMatrix<ST> A(rows, cols);
    SGVector<ST> b(cols);

    for (index_t i = 0; i < cols; ++i)
    {
        for (index_t j = 0; j < rows; ++j)
            A(j, i) = i * rows + j;
        b[i]= 2 * i;
    }

    auto x = matrix_prod(A, b);

    ST ref[] = {40, 46, 52, 58};

    EXPECT_EQ(x.vlen, A.num_rows);
    for (index_t i = 0; i < rows; ++i)
        EXPECT_NEAR(x[i], ref[i], 1e-15);
}

template <typename ST>
void check_SGVector_matrix_prod_transpose()
{
    const index_t rows=4;
    const index_t cols=3;

    SGMatrix<ST> A(cols, rows);
    SGVector<ST> b(cols);

    for (index_t i = 0; i < cols; ++i)
    {
        for (index_t j = 0; j < rows; ++j)
            A(i, j) = i * cols + j;
        b[i] = 2 * i;
    }

    auto x = matrix_prod(A, b, true);

    ST ref[] = {30, 36, 42};

    EXPECT_EQ(x.vlen, A.num_cols);
    for (index_t i = 0; i < cols; ++i)
        EXPECT_NEAR(x[i], ref[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_SGVector_matrix_prod_in_place()
{
    const index_t rows=4;
    const index_t cols=3;

    SGMatrix<ST> A(rows, cols);
    SGVector<ST> b(cols);
    SGVector<ST> x(rows);

    for (index_t i = 0; i<cols; ++i)
    {
        for (index_t j = 0; j < rows; ++j)
            A(j, i) = i * rows + j;
        b[i] = 2 * i;
    }

    matrix_prod(A, b, x);

    ST ref[] = {40, 46, 52, 58};

    for (index_t i = 0; i < cols; ++i)
        EXPECT_NEAR(x[i], ref[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_SGVector_matrix_prod_in_place_transpose()
{
    const index_t rows=4;
    const index_t cols=3;

    SGMatrix<ST> A(cols, rows);
    SGVector<ST> b(cols);
    SGVector<ST> x(rows);

    for (index_t i = 0; i < cols; ++i)
    {
        for (index_t j = 0; j < rows; ++j)
            A(i, j) = i * cols + j;
        b[i] = 2 * i;
    }

    matrix_prod(A, b, x, true);

    ST ref[] = {30, 36, 42};

    for (index_t i = 0; i < cols; ++i)
        EXPECT_NEAR(x[i], ref[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_matrix_product()
{
    const index_t dim1 = 2, dim2 = 4, dim3 = 2;
    SGMatrix<ST> A(dim1, dim2);
    SGMatrix<ST> B(dim2, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;

    auto cal = linalg::matrix_prod(A, B);

    ST ref[] = {28, 34, 76, 98};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_transpose_A()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim2, dim1);
    SGMatrix<ST> B(dim2, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;

    auto cal = linalg::matrix_prod(A, B, true);

    ST ref[] = {5, 14, 14, 50, 23, 86};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_transpose_B()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim1, dim2);
    SGMatrix<ST> B(dim3, dim2);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;

    auto cal = linalg::matrix_prod(A, B, false, true);

    ST ref[] = {30, 39, 36, 48, 42, 57};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_transpose_A_B()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim2, dim1);
    SGMatrix<ST> B(dim3, dim2);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;

    auto cal = linalg::matrix_prod(A, B, true, true);

    ST ref[] = {15, 42, 18, 54, 21, 66};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_in_place()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim1, dim2);
    SGMatrix<ST> B(dim2, dim3);
    SGMatrix<ST> cal(dim1, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;
    cal.zero();

    linalg::matrix_prod(A, B, cal);

    ST ref[] = {10, 13, 28, 40, 46, 67};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_in_place_transpose_A()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim2, dim1);
    SGMatrix<ST> B(dim2, dim3);
    SGMatrix<ST> cal(dim1, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;
    cal.zero();

    linalg::matrix_prod(A, B, cal, true);

    ST ref[] = {5, 14, 14, 50, 23, 86};


    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}


template <typename ST>
void check_SGMatrix_matrix_product_in_place_transpose_B()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim1, dim2);
    SGMatrix<ST> B(dim3, dim2);
    SGMatrix<ST> cal(dim1, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
        A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
        B[i] = i;
    cal.zero();

    linalg::matrix_prod(A, B, cal, false, true);

    ST ref[] = {30, 39, 36, 48, 42, 57};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
        EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGMatrix_matrix_product_in_place_transpose_A_B()
{
    const index_t dim1 = 2, dim2 = 3, dim3 = 3;
    SGMatrix<ST> A(dim2, dim1);
    SGMatrix<ST> B(dim3, dim2);
    SGMatrix<ST> cal(dim1, dim3);

    for (index_t i = 0; i < dim1*dim2; ++i)
    A[i] = i;
    for (index_t i = 0; i < dim2*dim3; ++i)
    B[i] = i;
    cal.zero();

    linalg::matrix_prod(A, B, cal, true, true);

    ST ref[] = {15, 42, 18, 54, 21, 66};

    EXPECT_EQ(dim1, cal.num_rows);
    EXPECT_EQ(dim3, cal.num_cols);
    for (index_t i = 0; i < dim1*dim3; ++i)
    EXPECT_EQ(ref[i], cal[i]);
}

template <typename ST>
void check_SGVector_max()
{
    SGVector<ST> A(9);

    ST a[] = {1, 2, 5, 8, 3, 1, 0, 2, 4};

    for (index_t i = 0; i < A.size(); ++i)
        A[i] = a[i];

    EXPECT_NEAR(8, max(A), 1e-15);
}

template <typename ST>
void check_SGMatrix_max()
{
    const index_t nrows = 3, ncols = 3;
    SGMatrix<ST> A(nrows, ncols);

    ST a[] = {1, 2, 5, 8, 3, 1, 0, 2, 12};

    for (index_t i = 0; i < nrows*ncols; ++i)
        A[i] = a[i];

    EXPECT_NEAR(12, max(A), 1e-15);
}

template <typename ST>
void check_SGVector_mean()
{
    const index_t size = 9;
    SGVector<ST> vec(size);
    vec.range_fill(0);

    auto result = mean(vec);

    EXPECT_NEAR(result, 4, 1E-15);
}

template <typename ST>
void check_SGMatrix_mean()
{
    const index_t nrows = 3, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);
    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    auto result = mean(mat);

    EXPECT_NEAR(result, 4, 1E-15);
}

template <typename ST>
void check_SGMatrix_multiply_by_logistic_derivative()
{
    SGMatrix<ST> A(3, 3);
    SGMatrix<ST> B(3, 3);

    for (ST i = 9; i < 9; i+=9)
    {
        A[i] = i / 9;
        B[i] = i;
    }

    linalg::multiply_by_logistic_derivative(A, B);

    for (index_t i = 0; i < 9; ++i)
        EXPECT_NEAR(i * A[i] * (1.0 - A[i]), B[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_multiply_by_rectified_linear_derivative()
{
    SGMatrix<ST> A(3, 3);
    SGMatrix<ST> B(3, 3);

    for (ST i = 0; i < 9; ++i)
    {
        A[i] = i * 0.5 - 0.5;
        B[i] = i;
    }

    multiply_by_rectified_linear_derivative(A, B);

    for (index_t i = 0; i < 9; ++i)
        EXPECT_NEAR(i * (A[i] != 0), B[i], 1e-15);
}

template <typename ST>
void check_SGVector_norm()
{
    const index_t n = 24; // evaluates to norm=70
    SGVector<ST> v(n);
    ST gt = 0;
    for (index_t i = 0; i < n; ++i)
    {
        v[i] = i;
        gt += i * i;
    }

    gt = std::sqrt(gt);

    auto result = norm(v);

    EXPECT_NEAR(result, gt, 1E-15);
}

template <typename ST>
void check_SGVector_qr_solver()
{
    const index_t n = 3;
    ST data_A[] = {0.02800922, 0.99326012, 0.15204902,
                   0.30492837, 0.39708534, 0.40466969,
                   0.36415317, 0.04407589, 0.9095746};
    ST data_b[] = {0.39461571, 0.6816856, 0.43323709};
    ST result[] = {0.07135206, 1.56393127, -0.23141312};

    SGMatrix<ST> A(data_A, n, n, false);
    SGVector<ST> b(data_b, n, false);

    auto x = qr_solver(A, b);

    for (index_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(x[i], result[i], 1E-6);
}

template <typename ST>
void check_SGMatrix_qr_solver()
{
    const index_t n = 3, m = 2;
    ST data_A[] = {0.02800922, 0.99326012, 0.15204902,
                   0.30492837, 0.39708534, 0.40466969,
                   0.36415317, 0.04407589, 0.9095746};
    ST data_B[] = {0.76775073, 0.88471312, 0.34795225,
                   0.94311546, 0.59630347, 0.65820143};
    ST result[] = {-0.73834587, 4.22750496, -1.37484721,
                   -1.14718091, 4.49142548, -1.08282992};

    SGMatrix<ST> A(data_A, n, n, false);
    SGMatrix<ST> B(data_B, n, m, false);

    auto X = qr_solver(A, B);

    for (index_t i = 0; i < (index_t)X.size(); ++i)
        EXPECT_NEAR(X[i], result[i], 1E-5);
}

template <typename ST>
void check_SGVector_range_fill()
{
    const index_t size = 5;
    SGVector<ST> vec(size);
    ST start = 1;  // FIXME: this is a bit awkward
    range_fill(vec, start);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(vec[i], i + 1, 1E-15);
}

template <typename ST>
void check_SGMatrix_range_fill()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);
    ST start = 1;
    range_fill(mat, start);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(mat[i], i + 1, 1E-15);
}

template <typename ST>
void check_SGMatrix_rectified_linear()
{
    SGMatrix<ST> A(3, 3);
    SGMatrix<ST> B(3, 3);
    ST start = 1;
    range_fill(A, start);

    linalg::rectified_linear(A, B);

    for (index_t i = 0; i < 9; ++i)
        EXPECT_NEAR(CMath::max(static_cast<ST>(0.0), A[i]), B[i], 1e-15);
}

template <typename ST>
void check_SGVector_scale()
{
    const index_t size = 5;
    const ST alpha = 2;
    SGVector<ST> a(size);
    a.range_fill(0);

    auto result = scale(a, alpha);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(alpha * a[i], result[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_scale()
{
    const ST alpha = 2;
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> A(nrows, ncols);

    for (index_t i = 0; i < nrows*ncols; ++i)
        A[i] = i;

    auto result = scale(A, alpha);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(alpha*A[i], result[i], 1e-15);
}

template <typename ST>
void check_SGVector_scale_in_place()
{
    const index_t size = 5;
    const ST alpha = 2;
    SGVector<ST> a(size);
    a.range_fill(0);

    scale(a, a, alpha);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(alpha * i, a[i], 1e-15);
}

template <typename ST>
void check_SGMatrix_scale_in_place()
{
    const ST alpha = 2;
    const index_t nrows = 2, ncols = 3;

    SGMatrix<ST> A(nrows, ncols);

    for (index_t i = 0; i < nrows*ncols; ++i)
        A[i] = i;

    scale(A, A, alpha);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(alpha*i, A[i], 1e-15);
}

template <typename ST>
void check_SGVector_set_const()
{
    const index_t size = 5;
    const ST value = 2;
    SGVector<ST> a(size);

    set_const(a, value);

    for (index_t i = 0; i < size; ++i)
        EXPECT_NEAR(a[i], value, 1E-15);
}

template <typename ST>
void check_SGMatrix_set_const()
{
    const index_t nrows = 2, ncols = 3;
    const ST value = 2;
    SGMatrix<ST> a(nrows, ncols);

    set_const(a, value);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_NEAR(a[i], value, 1E-15);
}


template <typename ST>
void check_SGMatrix_softmax()
{
    SGMatrix<ST> A(4, 3);
    SGMatrix<ST> ref(4, 3);

    for (ST i = 0; i < 12; ++i)
        A[i] = i / 12;

    for (index_t i = 0; i < 12; ++i)
        ref[i] = std::exp(A[i]);

    for (index_t j = 0; j < ref.num_cols; ++j)
    {
        ST sum = 0;
        for (index_t i = 0; i < ref.num_rows; ++i)
            sum += ref(i, j);

        for (index_t i = 0; i < ref.num_rows; ++i)
            ref(i, j) /= sum;
    }

    linalg::softmax(A);

    for (index_t i = 0; i < 12; ++i)
        EXPECT_NEAR(ref[i], A[i], 1e-7);
}

template <typename ST>
void check_SGMatrix_squared_error()
{
    SGMatrix<ST> A(4, 3);
    SGMatrix<ST> B(4, 3);

    ST size = A.num_rows * A.num_cols;
    for (ST i = 0; i < size; i+=2)
    {
        A[i] = i / size;
        B[i] = (i / size) * 2;
    }

    ST ref = 0;
    for (index_t i = 0; i < size; i++)
        ref += CMath::pow(A[i] - B[i], 2);
    ref *= 0.5;

    printf("%d", ref);

    auto result = linalg::squared_error(A, B);
        EXPECT_NEAR(ref, result, 1e-15);
}


template <typename ST>
void check_SGVector_sum()
{
    const index_t size = 10;
    SGVector<ST> vec(size);
    vec.range_fill(0);

    auto result = sum(vec);

    EXPECT_NEAR(result, 45, 1E-15);
}

template <typename ST>
void check_SGMatrix_sum()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    auto result = sum(mat);

    EXPECT_NEAR(result, 15, 1E-15);
}

template <typename ST>
void check_SGMatrix_sum_no_diag()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    auto result = sum(mat, true);

    EXPECT_NEAR(result, 12, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_with_diag()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n);
    mat.set_const(1);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = i + 1; j < n; ++j)
        {
            mat(i, j) = i * 10 + j + 1;
            mat(j, i) = mat(i, j);
        }

    EXPECT_NEAR(sum_symmetric(mat), 39, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_no_diag()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n);
    mat.set_const(1);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = i + 1; j < n; ++j)
        {
            mat(i, j) = i * 10 + j + 1;
            mat(j, i) = mat(i, j);
        }

    EXPECT_NEAR(sum_symmetric(mat, true), 36, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_exception()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n + 1);
    mat.set_const(1.0);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = i + 1; j < n; ++j)
        {
            mat(i, j) = i * 10 + j + 1;
            mat(j, i) = mat(i, j);
        }

    EXPECT_THROW(sum_symmetric(mat), ShogunException);
}

template <typename ST>
void check_SGMatrix_block_sum()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = 0; j < n; ++j)
            mat(i, j)=i * 10 + j + 1;

    auto result = sum(linalg::block(mat, 0, 0, 2, 3));
    EXPECT_NEAR(result, 42.0, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_block_with_diag()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n);
    mat.set_const(1);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = i + 1; j < n; ++j)
        {
            mat(i, j) = i * 10 + j + 1;
            mat(j, i) = mat(i, j);
        }

    ST sum = sum_symmetric(linalg::block(mat,1,1,2,2));
    EXPECT_NEAR(sum, 28, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_block_no_diag()
{
    const index_t n = 3;
    SGMatrix<ST> mat(n, n);
    mat.set_const(1);

    for (index_t i = 0; i < n; ++i)
        for (index_t j = i + 1; j < n; ++j)
        {
            mat(i, j) = i * 10 + j + 1;
            mat(j, i) = mat(i, j);
        }

    ST sum = sum_symmetric(linalg::block(mat,1,1,2,2), true);
    EXPECT_NEAR(sum, 26, 1E-15);
}

template <typename ST>
void check_SGMatrix_symmetric_block_exception()
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

template <typename ST>
void check_SGMatrix_colwise_sum()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    SGVector<ST> result = colwise_sum(mat);

    for (index_t j = 0; j < ncols; ++j)
    {
        ST sum = 0;
        for (index_t i = 0; i < nrows; ++i)
            sum += mat(i, j);
        EXPECT_NEAR(sum, result[j], 1E-15);
    }
}

template <typename ST>
void check_SGMatrix_colwise_sum_no_diag()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    SGVector<ST> result = colwise_sum(mat, true);

    EXPECT_NEAR(result[0], 1, 1E-15);
    EXPECT_NEAR(result[1], 2, 1E-15);
    EXPECT_NEAR(result[2], 9, 1E-15);
}

template <typename ST>
void check_SGMatrix_block_colwise_sum()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<float64_t> mat(nrows, ncols);

    for (index_t i = 0; i < nrows; ++i)
        for (index_t j = 0; j < ncols; ++j)
            mat(i, j) = i * 10 + j + 1;

    auto result = colwise_sum(linalg::block(mat, 0, 0, 2, 3));

    for (index_t j = 0; j < ncols; ++j)
    {
        ST sum = 0;
        for (index_t i = 0; i < nrows; ++i)
            sum += mat(i, j);
        EXPECT_NEAR(sum, result[j], 1E-15);
    }
}


template <typename ST>
void check_SGMatrix_rowwise_sum()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    SGVector<ST> result = rowwise_sum(mat);

    for (index_t i = 0; i < nrows; ++i)
    {
        ST sum = 0;
        for (index_t j = 0; j < ncols; ++j)
            sum += mat(i, j);
        EXPECT_NEAR(sum, result[i], 1E-15);
    }
}

template <typename ST>
void check_SGMatrix_rowwise_sum_no_diag()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows * ncols; ++i)
        mat[i] = i;

    SGVector<ST> result = rowwise_sum(mat, true);

    EXPECT_NEAR(result[0], 6, 1E-15);
    EXPECT_NEAR(result[1], 6, 1E-15);
}

template <typename ST>
void check_SGMatrix_block_rowwise_sum()
{
    const index_t nrows = 2, ncols = 3;
    SGMatrix<ST> mat(nrows, ncols);

    for (index_t i = 0; i < nrows; ++i)
        for (index_t j = 0; j < ncols; ++j)
            mat(i, j) = i * 10 + j + 1;

    auto result = rowwise_sum(linalg::block(mat, 0, 0, 2, 3));

    for (index_t i = 0; i < nrows; ++i)
    {
        ST sum = 0;
        for (index_t j = 0; j < ncols; ++j)
            sum += mat(i, j);
        EXPECT_NEAR(sum, result[i], 1E-15);
    }
}

template <typename ST>
void check_SGMatrix_svd_jacobi_thinU()
{
    const index_t m = 5, n = 3;
    ST data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
                 0.30786772, 0.25503552, 0.34367041, 0.66491478,
                 0.20488809, 0.5734351,  0.87179189, 0.07139643,
                 0.28540373, 0.06264684, 0.56204061};
    ST result_s[] = {1.75382524, 0.56351367, 0.41124883};
    ST result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
                     -0.46186125, -0.69145782, 0.29548428,  0.5718984,
                     0.31771648,  -0.08101592, -0.27461424, 0.37170223,
                     -0.12681555, -0.53830325, 0.69323293};

    SGMatrix<ST> A(data, m, n, false);
    SGMatrix<ST> U(m, n);
    SGVector<ST> s(n);

    svd(A, s, U, true, SVDAlgorithm::Jacobi);

    for (index_t i = 0; i < n; ++i)
    {
        auto c = CMath::sign(U[i * m] * result_U[i * m]);
        for (index_t j = 0; j < m; ++j)
            EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
    }
    for (index_t i = 0; i < (index_t)s.size(); ++i)
        EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

template <typename ST>
void check_SGMatrix_svd_jacobi_fullU()
{
    const index_t m = 5, n = 3;
    ST data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
                 0.30786772, 0.25503552, 0.34367041, 0.66491478,
                 0.20488809, 0.5734351,  0.87179189, 0.07139643,
                 0.28540373, 0.06264684, 0.56204061};
    ST result_s[] = {1.75382524, 0.56351367, 0.41124883};
    ST result_U[] = {
            -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
            -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
            -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
            -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
            0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

    SGMatrix<ST> A(data, m, n, false);
    SGMatrix<ST> U(m, m);
    SGVector<ST> s(n);

    svd(A, s, U, false, SVDAlgorithm::Jacobi);

    for (index_t i = 0; i < n; ++i)
    {
        auto c = CMath::sign(U[i * m] * result_U[i * m]);
        for (index_t j = 0; j < m; ++j)
            EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
    }
    for (index_t i = 0; i < (index_t)s.size(); ++i)
        EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
template <typename ST>
void check_SGMatrix_svd_bdc_thinU()
{
    const index_t m = 5, n = 3;
    ST data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
                 0.30786772, 0.25503552, 0.34367041, 0.66491478,
                 0.20488809, 0.5734351,  0.87179189, 0.07139643,
                 0.28540373, 0.06264684, 0.56204061};
    ST result_s[] = {1.75382524, 0.56351367, 0.41124883};
    ST result_U[] = {-0.60700926, -0.16647013, -0.56501385, -0.26696629,
                     -0.46186125, -0.69145782, 0.29548428,  0.5718984,
                     0.31771648,  -0.08101592, -0.27461424, 0.37170223,
                     -0.12681555, -0.53830325, 0.69323293};

    SGMatrix<ST> A(data, m, n, false);
    SGMatrix<ST> U(m, n);
    SGVector<ST> s(n);

    svd(A, s, U, true, SVDAlgorithm::BidiagonalDivideConquer);

    for (index_t i = 0; i < n; ++i)
    {
        auto c = CMath::sign(U[i * m] * result_U[i * m]);
        for (index_t j = 0; j < m; ++j)
            EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
    }
    for (index_t i = 0; i < (index_t)s.size(); ++i)
        EXPECT_NEAR(s[i], result_s[i], 1e-6);
}

template <typename ST>
void check_SGMatrix_svd_bdc_fullU()
{
    const index_t m = 5, n = 3;
    ST data[] = {0.68764958, 0.11456779, 0.75164207, 0.50436194,
                 0.30786772, 0.25503552, 0.34367041, 0.66491478,
                 0.20488809, 0.5734351,  0.87179189, 0.07139643,
                 0.28540373, 0.06264684, 0.56204061};
    ST result_s[] = {1.75382524, 0.56351367, 0.41124883};
    ST result_U[] = {
            -0.60700926, -0.16647013, -0.56501385, -0.26696629, -0.46186125,
            -0.69145782, 0.29548428,  0.5718984,   0.31771648,  -0.08101592,
            -0.27461424, 0.37170223,  -0.12681555, -0.53830325, 0.69323293,
            -0.27809756, -0.68975171, -0.11662812, 0.38274703,  0.53554354,
            0.025973184, 0.520631112, -0.56921636, 0.62571522,  0.11287970};

    SGMatrix<ST> A(data, m, n, false);
    SGMatrix<ST> U(m, m);
    SGVector<ST> s(n);

    svd(A, s, U, false, SVDAlgorithm::BidiagonalDivideConquer);

    for (index_t i = 0; i < n; ++i)
    {
        auto c = CMath::sign(U[i * m] * result_U[i * m]);
        for (index_t j = 0; j < m; ++j)
            EXPECT_NEAR(U[i * m + j], c * result_U[i * m + j], 1e-6);
    }
    for (index_t i = 0; i < (index_t)s.size(); ++i)
        EXPECT_NEAR(s[i], result_s[i], 1e-6);
}
#endif

template <typename ST>
void check_SGMatrix_trace()
{
    const index_t n = 4;

    SGMatrix<ST> A(n, n);
    for (index_t i = 0; i < n*n; ++i)
        A[i] = i;

    ST tr = 0;
    for (index_t i = 0; i < n; ++i)
        tr += A.get_element(i, i);

    EXPECT_NEAR(trace(A), tr, 1e-15);
}

template <typename ST>
void check_SGMatrix_trace_dot()
{
    const index_t n = 2;
    SGMatrix<ST> A(n, n), B(n, n);
    for (index_t i = 0; i < n*n; ++i) {
        A[i] = i;
        B[i] = i * 2;
    }

    auto C = matrix_prod(A, B);
    auto tr = 0.0;
    for (auto i : range(n))
        tr += C(i, i);

    EXPECT_NEAR(tr, trace_dot(A, B), 1e-15);
}

template <typename ST>
void check_SGMatrix_transpose_matrix()
{
    const index_t m = 5, n = 3;
    SGMatrix<ST> A(m, n);
    linalg::range_fill(A, static_cast<ST>(1));

    auto T = transpose_matrix(A);

    for (index_t i = 0; i < m; ++i)
        for (index_t j = 0; j < n; ++j)
            EXPECT_NEAR(A.get_element(i, j), T.get_element(j, i), 1e-15);
}

template <typename ST>
void check_SGVector_triangular_solver_lower()
{
    const index_t n = 3;
    ST data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
                   0.,          -0.27048649, -0.05919915,
                   0.,          0.,          0.11869106};
    ST data_b[] = {0.39461571, 0.6816856, 0.43323709};
    ST result[] = {-0.42455592, -0.72571316, 0.17192745};

    SGMatrix<ST> L(data_L, n, n, false);
    SGVector<ST> b(data_b, n, false);

    auto x = triangular_solver(L, b, true);

    for (index_t i = 0; i < (index_t)x.size(); ++i)
        EXPECT_NEAR(x[i], result[i], 1E-6);
}

template <typename ST>
void check_SGVector_triangular_solver_upper()
{
    const index_t n = 3;
    ST data_U[] = {-0.92947874, 0.,          0.,
                   -1.1432887,  -0.27048649, 0.,
                   -0.87119086, -0.05919915, 0.11869106};
    ST data_b[] = {0.39461571, 0.6816856, 0.43323709};
    ST result[] = {0.23681135, -3.31909306, 3.65012412};

    SGMatrix<ST> U(data_U, n, n, false);
    SGVector<ST> b(data_b, n, false);

    auto x = triangular_solver(U, b, false);

    for (index_t i = 0; i < (index_t)x.size(); ++i)
        EXPECT_NEAR(x[i], result[i], 1E-6);
}

template <typename ST>
void check_SGMatrix_triangular_solver_lower()
{
    const index_t n = 3, m = 2;
    ST data_L[] = {-0.92947874, -1.1432887,  -0.87119086,
                   0.,          -0.27048649, -0.05919915,
                   0.,          0.,          0.11869106};
    ST data_B[] = {0.76775073, 0.88471312, 0.34795225,
                   0.94311546, 0.59630347, 0.65820143};
    ST result[] = {-0.82600139, 0.22050986, -3.02127745,
                   -1.01467136, 2.08424024, -0.86262387};

    SGMatrix<ST> L(data_L, n, n, false);
    SGMatrix<ST> B(data_B, n, m, false);

    auto X = triangular_solver(L, B, true);

    for (index_t i = 0; i < (index_t)X.size(); ++i)
    EXPECT_NEAR(X[i], result[i], 1E-6);
}

template <typename ST>
void check_SGMatrix_triangular_solver_upper()
{
    const index_t n = 3, m = 2;
    ST data_U[] = {-0.92947874, 0.,          0.,
                   -1.1432887,  -0.27048649, 0.,
                   -0.87119086, -0.05919915, 0.11869106};
    ST data_B[] = {0.76775073, 0.88471312, 0.34795225,
                   0.94311546, 0.59630347, 0.65820143};
    ST result[] = {1.238677,    -3.91243241, 2.9315793,
                   -2.00784647, -3.41825732, 5.54550138};

    SGMatrix<ST> L(data_U, n, n, false);
    SGMatrix<ST> B(data_B, n, m, false);

    auto X = triangular_solver(L, B, false);

    for (index_t i = 0; i < (index_t)X.size(); ++i)
    EXPECT_NEAR(X[i], result[i], 1E-6);
}


template <typename ST>
void check_SGVector_zero()
{
    const index_t n = 16;
    SGVector<ST> a(n);
    zero(a);

    for (index_t i = 0; i < n; ++i)
        EXPECT_EQ(a[i], 0);
}

template <typename ST>
void check_SGMatrix_zero()
{
    const index_t nrows = 3, ncols = 4;
    SGMatrix<ST> A(nrows, ncols);
    zero(A);

    for (index_t i = 0; i < nrows*ncols; ++i)
        EXPECT_EQ(A[i], 0);
}

template <typename ST>
void check_SGMatrix_rank_update()
{
    typedef Matrix<ST, Dynamic, Dynamic> Mxx;
    typedef Matrix<ST, 1, Dynamic> Vxd;

    const index_t size = 2;
    SGMatrix<ST> A(size, size);
    SGVector<ST> b(size);
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

    rank_update(A, b, static_cast<ST>(1));
    EXPECT_NEAR((A2_eig - A_eig).norm(), 0, 1e-14);

    rank_update(A, b, static_cast<ST>(-1));
    EXPECT_NEAR((A_eig - A_eig).norm(), 0, 1e-14);
}


// test types based on shogun/mathematics/linalg/LinalgBackendBase.h
TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add)
{
    check_SGVector_add<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add)
{
    check_SGMatrix_add<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_in_place)
{
    check_SGVector_add_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_in_place)
{
    check_SGMatrix_add_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_col_vec_allocated)
{
    check_SGVector_add_col_vec_allocated<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_col_vec_in_place)
{
    check_SGVector_add_col_vec_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_col_vec_allocated)
{
    check_SGMatrix_add_col_vec_allocated<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_col_vec_in_place)
{
    check_SGMatrix_add_col_vec_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_diag)
{
    check_add_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_ridge)
{
    check_add_ridge<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, add_vector)
{
    check_add_vector<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_add_scalar)
{
    check_SGVector_add_scalar<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_add_scalar)
{
    check_SGMatrix_add_scalar<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_center_matrix)
{
    check_SGMatrix_center_matrix<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_llt_lower)
{
    check_SGMatrix_cholesky_llt_lower<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_llt_upper)
{
    check_SGMatrix_cholesky_llt_upper<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_rank_update_upper)
{
    check_SGMatrix_cholesky_rank_update_upper<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_rank_update_lower)
{
    check_SGMatrix_cholesky_rank_update_lower<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cholesky_ldlt_lower)
{
    check_SGMatrix_cholesky_ldlt_lower<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenRealTypesTest, SGMatrix_cholesky_solver)
{
    check_SGMatrix_cholesky_solver<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_ldlt_solver)
{
    check_SGMatrix_ldlt_solver<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_cross_entropy)
{
    check_SGMatrix_cross_entropy<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_psd)
{
    check_SGMatrix_pinv_psd<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_2x4)
{
    check_SGMatrix_pinv_2x4<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_pinv_4x2)
{
    check_SGMatrix_pinv_4x2<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_dot)
{
    check_SGVector_dot<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, eigensolver)
{
    check_eigensolver<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, eigensolver_symmetric)
{
    check_eigensolver_symmetric<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_elementwise_product)
{
    check_SGMatrix_elementwise_product<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_elementwise_product_in_place)
{
    check_SGMatrix_elementwise_product_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_elementwise_product)
{
    check_SGMatrix_block_elementwise_product<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_elementwise_product)
{
    check_SGVector_elementwise_product<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_elementwise_product_in_place)
{
    check_SGVector_elementwise_product_in_place<TypeParam>();
}

// TODO: write test for int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_exponent)
{
    check_SGVector_exponent<TypeParam>();
}

// TODO: write test for int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_exponent)
{
    check_SGMatrix_exponent<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_identity)
{
    check_SGMatrix_identity<TypeParam>();
}

// TODO: write test for int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, logistic)
{
    check_logistic<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_SGVector_matrix_prod)
{
    check_SGMatrix_SGVector_matrix_prod<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_matrix_prod_transpose)
{
    check_SGVector_matrix_prod_transpose<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_SGVector_matrix_prod_in_place)
{
    check_SGMatrix_SGVector_matrix_prod_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_SGVector_matrix_prod_in_place_transpose)
{
    check_SGMatrix_SGVector_matrix_prod_in_place_transpose<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product)
{
    check_SGMatrix_matrix_product<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_A)
{
    check_SGMatrix_matrix_product_transpose_A<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_B)
{
    check_SGMatrix_matrix_product_transpose_B<TypeParam>();
}


TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_transpose_A_B)
{
    check_SGMatrix_matrix_product_transpose_A_B<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_in_place)
{
    check_SGMatrix_matrix_product_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_in_place_transpose_A)
{
    check_SGMatrix_matrix_product_in_place_transpose_A<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_in_place_transpose_B)
{
    check_SGMatrix_matrix_product_in_place_transpose_B<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_matrix_product_in_place_transpose_A_B)
{
    check_SGMatrix_matrix_product_in_place_transpose_A_B<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_max)
{
    check_SGVector_max<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_max)
{
    check_SGMatrix_max<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_mean)
{
    check_SGVector_mean<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_mean)
{
    check_SGMatrix_mean<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_multiply_by_logistic_derivative)
{
    check_SGMatrix_multiply_by_logistic_derivative<TypeParam>();
}

// TODO: write test for int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_multiply_by_rectified_linear_derivative)
{
    check_SGMatrix_multiply_by_rectified_linear_derivative<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_norm)
{
    check_SGVector_norm<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_qr_solver)
{
    check_SGVector_qr_solver<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_qr_solver)
{
    check_SGMatrix_qr_solver<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_range_fill)
{
    check_SGVector_range_fill<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_range_fill)
{
    check_SGMatrix_range_fill<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rectified_linear)
{
    check_SGMatrix_rectified_linear<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_scale)
{
    check_SGVector_scale<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_scale)
{
    check_SGMatrix_scale<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_scale_in_place)
{
    check_SGVector_scale_in_place<TypeParam>();
}


TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_scale_in_place)
{
    check_SGMatrix_scale_in_place<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_set_const)
{
    check_SGVector_set_const<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_set_const)
{
    check_SGMatrix_set_const<TypeParam>();
}

// TODO: extend to all types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_softmax)
{
    check_SGMatrix_softmax<TypeParam>();
}

// FIXME: CMath::Pow only accepts float or complex types
//TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_squared_error)
//{
//    check_SGMatrix_squared_error<TypeParam>();
//}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_sum)
{
    check_SGVector_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_sum)
{
    check_SGMatrix_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_sum_no_diag)
{
    check_SGMatrix_sum_no_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_with_diag)
{
    check_SGMatrix_symmetric_with_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_no_diag)
{
    check_SGMatrix_symmetric_no_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_exception)
{
    check_SGMatrix_symmetric_exception<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_sum)
{
    check_SGMatrix_block_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_with_diag)
{
    check_SGMatrix_symmetric_block_with_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_no_diag)
{
    check_SGMatrix_symmetric_block_no_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_symmetric_block_exception)
{
    check_SGMatrix_symmetric_block_exception<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_colwise_sum)
{
    check_SGMatrix_colwise_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_colwise_sum_no_diag)
{
    check_SGMatrix_colwise_sum_no_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_colwise_sum)
{
    check_SGMatrix_block_colwise_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rowwise_sum)
{
    check_SGMatrix_rowwise_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_rowwise_sum_no_diag)
{
    check_SGMatrix_rowwise_sum_no_diag<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_block_rowwise_sum)
{
    check_SGMatrix_block_rowwise_sum<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_jacobi_thinU)
{
    check_SGMatrix_svd_jacobi_thinU<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_jacobi_fullU)
{
    check_SGMatrix_svd_jacobi_fullU<TypeParam>();
}


#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_bdc_thinU)
{
    check_SGMatrix_svd_bdc_thinU<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_svd_bdc_fullU)
{
    check_SGMatrix_svd_bdc_fullU<TypeParam>();
}
#endif

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_trace)
{
    check_SGMatrix_trace<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_trace_dot)
{
    check_SGMatrix_trace_dot<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_transpose_matrix)
{
    check_SGMatrix_transpose_matrix<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_triangular_solver_lower)
{
    check_SGVector_triangular_solver_lower<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGVector_triangular_solver_upper)
{
    check_SGVector_triangular_solver_upper<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_triangular_solver_lower)
{
    check_SGMatrix_triangular_solver_lower<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_triangular_solver_upper)
{
    check_SGMatrix_triangular_solver_upper<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGVector_zero)
{
    check_SGVector_zero<TypeParam>();
}

TYPED_TEST(LinalgBackendEigenAllTypesTest, SGMatrix_zero)
{
    check_SGMatrix_zero<TypeParam>();
}

// TODO: extend to int types
TYPED_TEST(LinalgBackendEigenNonIntegerTypesTest, SGMatrix_rank_update)
{
    check_SGMatrix_rank_update<TypeParam>();
}