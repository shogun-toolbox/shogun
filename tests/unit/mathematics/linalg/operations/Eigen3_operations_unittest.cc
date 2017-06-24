#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgSpecialPurposes.h>
#include <gtest/gtest.h>
#include <shogun/lib/ShogunException.h>

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
		EXPECT_NEAR(result.get_element(i, col),
		            alpha*A.get_element(i, col)+beta*b[i], 1e-15);
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

TEST(LinalgBackendEigen, SGMatrix_QR_solver)
{
	const index_t size=2;
	SGMatrix<float64_t> A(size, size);
	A(0,0)=2.0;
	A(0,1)=1.0;
	A(1,0)=1.0;
	A(1,1)=2.5;

	SGMatrix<float64_t> b(size, size);
	b(0,0) = 10;
	b(0,1) = 13;
	b(1,0) = 10;
	b(1,1) = 13;

	SGMatrix<float64_t> x_ref(size, size);
	x_ref(0,0) = 3;
	x_ref(0,1) = 4;
	x_ref(1,0) = 3;
	x_ref(1,1) = 4;

	SGMatrix<float64_t> x_cal = qr_solver(A, b);

	EXPECT_NEAR(x_ref(0,0), x_cal(0,0), 1E-15);
	EXPECT_NEAR(x_ref(0,1), x_cal(0,1), 1E-15);
	EXPECT_EQ(x_ref.size(), x_cal.size());
}

TEST(LinalgBackendEigen, SGMatrix_QR_decomposition)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	SGMatrix<float64_t> Q = cholesky_factor(m, false);
	SGMatrix<float64_t> R = cholesky_factor(m);

	Map<MatrixXd> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<MatrixXd> map_Q(Q.matrix,Q.num_rows,Q.num_cols);
	Map<MatrixXd> map_R(R.matrix,R.num_rows,R.num_cols);
	EXPECT_NEAR((map_A-map_Q*map_R).norm(),0.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_SVD_solver)
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

	SGVector<float64_t> x_cal = svd_solver(A, b);

	EXPECT_NEAR(x_ref[0], x_cal[0], 1E-15);
	EXPECT_NEAR(x_ref[1], x_cal[1], 1E-15);
	EXPECT_EQ(x_ref.size(), x_cal.size());
}

TEST(LinalgBackendEigen, SGMatrix_SVD_decomposition)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	SGMatrix<float64_t> S = svd_s(m);
	SGMatrix<float64_t> V = svd_v(m);
	SGMatrix<float64_t> U = svd_u(m);

	Map<MatrixXd> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<MatrixXd> map_S(S.matrix,S.num_rows,S.num_cols);
	Map<MatrixXd> map_V(V.matrix,V.num_rows,V.num_cols);
	Map<MatrixXd> map_U(U.matrix,U.num_rows,U.num_cols);
	EXPECT_NEAR((map_A-map_U*map_S*map_V.transpose()).norm(),0.0, 1E-15);
}

TEST(LinalgBackendEigen, SGMatrix_svd_solver)
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

	SGVector<float64_t> x_cal = svd_solver(A, b);

	EXPECT_NEAR(x_ref[0], x_cal[0], 1E-15);
	EXPECT_NEAR(x_ref[1], x_cal[1], 1E-15);
	EXPECT_EQ(x_ref.size(), x_cal.size());
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

TEST(LinalgBackendEigen, SGMatrix_block_elementwise_product)
{
	const index_t nrows = 2;
	const index_t ncols = 3;

	SGMatrix<float64_t> A(nrows,ncols);
	SGMatrix<float64_t> B(ncols,nrows);

	for (index_t i = 0; i < nrows; ++i)
		for (index_t j = 0; j < ncols; ++j)
		{
			A(i, j) = i * 10 + j + 1;
			B(j, i) = i + j;
		}

	auto A_block = linalg::block(A, 0, 0, 2, 2);
	auto B_block = linalg::block(B, 0, 0, 2, 2);
	auto result = element_prod(A_block, B_block);

	ASSERT_EQ(result.num_rows, 2);
	ASSERT_EQ(result.num_cols, 2);

	for (index_t i = 0; i < 2; ++i)
		for (index_t j = 0; j < 2; ++j)
			EXPECT_NEAR(result(i, j), A(i, j) * B(i, j), 1E-15);
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
