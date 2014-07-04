#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif

using namespace shogun;

TEST(SGMatrixTest,ctor_zero_const)
{
	SGMatrix<float64_t> a(10, 5);
	EXPECT_EQ(a.num_rows, 10);
	EXPECT_EQ(a.num_cols, 5);

	a.zero();
	for (int i=0; i < 10; ++i)
	{
		for (int j=0; j < 5; ++j)
		EXPECT_EQ(0, a(i,j));
	}

	a.set_const(3.3);
	for (int i=0; i < 10; ++i)
	{
		for (int j=0; j < 5; ++j)
		EXPECT_EQ(3.3, a(i,j));
	}
}

TEST(SGMatrixTest,setget)
{
	SGMatrix<index_t> v(3,2);
	v(0,0)=1;
	v(0,1)=2;
	v(1,0)=3;
	v(1,1)=4;
	v(2,0)=5;
	v(2,1)=6;

	EXPECT_EQ(v(0,0), 1);
	EXPECT_EQ(v(0,1), 2);
	EXPECT_EQ(v(1,0), 3);
	EXPECT_EQ(v(1,1), 4);
	EXPECT_EQ(v(2,0), 5);
	EXPECT_EQ(v(2,1), 6);
}

TEST(SGMatrixTest,equals_equal)
{
	SGMatrix<float64_t> a(3,2);
	SGMatrix<float64_t> b(3,2);
	a(0,0)=1;
	a(0,1)=2;
	a(1,0)=3;
	a(1,1)=4;
	a(2,0)=5;
	a(2,1)=6;

	b(0,0)=1;
	b(0,1)=2;
	b(1,0)=3;
	b(1,1)=4;
	b(2,0)=5;
	b(2,1)=6;

	EXPECT_TRUE(a.equals(b));
}

TEST(SGMatrixTest,equals_different)
{
	SGMatrix<float64_t> a(3,2);
	SGMatrix<float64_t> b(3,2);
	a(0,0)=1;
	a(0,1)=2;
	a(1,0)=3;
	a(1,1)=4;
	a(2,0)=5;
	a(2,1)=6;

	b(0,0)=1;
	b(0,1)=2;
	b(1,0)=3;
	b(1,1)=4;
	b(2,0)=5;
	b(2,1)=7;

	EXPECT_FALSE(a.equals(b));
}

TEST(SGMatrixTest,equals_different_size)
{
	SGMatrix<float64_t> a(3,2);
	SGMatrix<float64_t> b(2,2);
	a.zero();
	b.zero();

	EXPECT_FALSE(a.equals(b));
}

TEST(SGMatrixTest,get_diagonal_vector_square_matrix)
{
	SGMatrix<int32_t> mat(3, 3);

	mat(0,0)=8;
	mat(0,1)=1;
	mat(0,2)=6;
	mat(1,0)=3;
	mat(1,1)=5;
	mat(1,2)=7;
	mat(2,0)=4;
	mat(2,1)=9;
	mat(2,2)=2;

	SGVector<int32_t> diag=mat.get_diagonal_vector();

	EXPECT_EQ(diag[0], 8);
	EXPECT_EQ(diag[1], 5);
	EXPECT_EQ(diag[2], 2);
}

TEST(SGMatrixTest,get_diagonal_vector_rectangular_matrix)
{
	SGMatrix<int32_t> mat(3, 2);

	mat(0,0)=8;
	mat(0,1)=1;
	mat(1,0)=3;
	mat(1,1)=5;
	mat(2,0)=4;
	mat(2,1)=9;

	SGVector<int32_t> diag=mat.get_diagonal_vector();

	EXPECT_EQ(diag[0], 8);
	EXPECT_EQ(diag[1], 5);
}

TEST(SGMatrixTest,is_symmetric_float32_false_old_plus_eps)
{
	const index_t size=2;
	SGMatrix<float32_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_float();
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			float32_t old_val=mat(i, j);
			float32_t diff=FLT_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_float32_false_old_minus_eps)
{
	const index_t size=2;
	SGMatrix<float32_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_float();
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			float32_t old_val=mat(i, j);
			float32_t diff=-FLT_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_float32_true)
{
	const index_t size=2;
	SGMatrix<float32_t> mat(size, size);
	CMath::init_random(100);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_float();
			mat(j, i)=mat(i, j);
		}
	}
	EXPECT_TRUE(mat.is_symmetric());
}

TEST(SGMatrixTest,is_symmetric_float64_false_old_plus_eps)
{
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_double();
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			float64_t old_val=mat(i, j);
			float64_t diff=DBL_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_float64_false_old_minus_eps)
{
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_double();
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			float64_t old_val=mat(i, j);
			float64_t diff=-DBL_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_float64_true)
{
	const index_t size=2;
	SGMatrix<float64_t> mat(size, size);
	CMath::init_random(100);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=CMath::randn_double();
			mat(j, i)=mat(i, j);
		}
	}
	EXPECT_TRUE(mat.is_symmetric());
}

TEST(SGMatrixTest,is_symmetric_complex128_false_old_plus_eps)
{
	const index_t size=2;
	SGMatrix<complex128_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=complex128_t(CMath::randn_double(), CMath::randn_double());
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			complex128_t old_val=mat(i, j);
			float64_t diff=DBL_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			mat(i, j)=old_val+complex128_t(0, diff);
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;

			mat(j, i)=old_val+complex128_t(0, diff);
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_complex128_false_old_minus_eps)
{
	const index_t size=2;
	SGMatrix<complex128_t> mat(size, size);
	CMath::init_random(100);

	// create a symmetric matrix
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=complex128_t(CMath::randn_double(), CMath::randn_double());
			mat(j, i)=mat(i, j);
		}
	}

	// modify one element in the matrix to destroy symmetry
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			complex128_t old_val=mat(i, j);
			float64_t diff=-DBL_EPSILON;

			// update, check, restore
			mat(i, j)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			mat(i, j)=old_val+complex128_t(0, diff);
			EXPECT_FALSE(mat.is_symmetric());
			mat(i, j)=old_val;

			// symmetric counterpart
			mat(j, i)=old_val+diff;
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;

			mat(j, i)=old_val+complex128_t(0, diff);
			EXPECT_FALSE(mat.is_symmetric());
			mat(j, i)=old_val;
		}
	}
}

TEST(SGMatrixTest,is_symmetric_complex128_true)
{
	const index_t size=2;
	SGMatrix<complex128_t> mat(size, size);
	CMath::init_random(100);
	for (index_t i=0; i<size; ++i)
	{
		for (index_t j=i+1; j<size; ++j)
		{
			mat(i, j)=complex128_t(CMath::randn_double(), CMath::randn_double());
			mat(j, i)=mat(i, j);
		}
	}
	EXPECT_TRUE(mat.is_symmetric());
}

#ifdef HAVE_EIGEN3

TEST(SGMatrixTest, to_eigen3)
{
	const int nrows = 3;
	const int ncols = 4;
	
	SGMatrix<float64_t> sg_mat(nrows,ncols);
 	for (int32_t i=0; i<nrows*ncols; i++)
 		sg_mat[i] = i;
	
	Eigen::Map<Eigen::MatrixXd> eigen_mat = sg_mat;
	
	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(sg_mat[i], eigen_mat(i));
}
 
TEST(SGMatrixTest, from_eigen3)
{
	const int nrows = 3;
	const int ncols = 4;
	
	Eigen::MatrixXd eigen_mat(nrows,ncols);
	for (int32_t i=0; i<nrows*ncols; i++)
		eigen_mat(i) = i;
	
	SGMatrix<float64_t> sg_mat = eigen_mat;
	
	for (int32_t i=0; i<nrows*ncols; i++)
		EXPECT_EQ(eigen_mat(i), sg_mat[i]);
}

#endif
