/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <gtest/gtest.h>

using namespace shogun;

using namespace Eigen;

TEST(DenseMatrixOperator, apply)
{
	const index_t size=5;
	SGVector<float64_t> b1(size);
	b1.set_const(0.25);

	// float64_t, fixed matrix
	SGMatrix<float64_t> m1(size, size);
	m1.set_const(0.5);

	CDenseMatrixOperator<float64_t> op11(m1);
	SGVector<float64_t> r1=op11.apply(b1);

	for (index_t i=0; i<r1.vlen; ++i)
	{
		EXPECT_NEAR(r1[i], 0.625, 1E-16);
	}

	// float64_t, identity matrix
	typedef Matrix<float64_t, size, size> MatrixSd;
	Map<MatrixSd> mapped1(m1.matrix, m1.num_rows, m1.num_cols);
	mapped1=MatrixSd::Identity();
	CDenseMatrixOperator<float64_t> op12(m1);
	r1=op12.apply(b1);

	for (index_t i=0; i<r1.vlen; ++i)
	{
		EXPECT_NEAR(r1[i], 0.25, 1E-16);
	}

	SGVector<complex128_t> b2(size);
	b2.set_const(complex128_t(0.25, 1.0));

	// complex128_t, fixed matrix
	SGMatrix<complex128_t> m2(size, size);
	m2.set_const(complex128_t(0.5, 0.25));

	CDenseMatrixOperator<complex128_t> op21(m2);
	SGVector<complex128_t> r2=op21.apply(b2);

	for (index_t i=0; i<r2.vlen; ++i)
	{
		EXPECT_NEAR(r2[i].real(), -0.625, 1E-16);
		EXPECT_NEAR(r2[i].imag(), 2.8125, 1E-16);
	}

	// complex128_t, identity matrix
	typedef Matrix<complex128_t, size, size> MatrixScd;
	Map<MatrixScd> mapped2(m2.matrix, m2.num_rows, m2.num_cols);
	mapped2=MatrixScd::Identity();
	CDenseMatrixOperator<complex128_t> op22(m2);
	r2=op22.apply(b2);

	for (index_t i=0; i<r2.vlen; ++i)
	{
		EXPECT_NEAR(r2[i].real(), 0.25, 1E-16);
		EXPECT_NEAR(r2[i].imag(), 1.0, 1E-16);
	}
}

TEST(DenseMatrixOperator, shift_apply)
{
	const index_t size=5;
	SGVector<complex128_t> b(size);
	b.set_const(0.25);

	// complex128_t, fixed matrix
	SGMatrix<complex128_t> m(size, size);
	m.set_const(complex128_t(0.5, 0.0));

	// shifting the diagonal via interface
	CDenseMatrixOperator<complex128_t> op(m);
	SGVector<complex128_t> diag=op.get_diagonal();
	for (index_t i=0; i<diag.vlen; ++i)
	{
#if defined(HAVE_CXX11) || defined(_LIBCPP_VERSION)
		diag[i].imag(diag[i].imag()-0.75);
#else
		diag[i].imag()-=0.75;
#endif
	}
	op.set_diagonal(diag);

	SGVector<complex128_t> r1=op.apply(b);
	Map<VectorXcd> map_r1(r1.vector, r1.vlen);

	// shifting the diagonal directly via matrix
	m.set_const(complex128_t(0.5, 0.0));
	for (index_t i=0; i<size; ++i)
	{
#if defined(HAVE_CXX11) || defined(_LIBCPP_VERSION)
		m(i,i).imag(m(i,i).imag()-0.75);
#else
		m(i,i).imag()-=0.75;
#endif
	}

	SGVector<complex128_t> r2=op.apply(b);
	Map<VectorXcd> map_r2(r2.vector, r2.vlen);

	EXPECT_NEAR(map_r1.norm(), map_r2.norm(), 1E-15);
}

TEST(DenseMatrixOperator, asymmetric_clone)
{
	const index_t length_x=2;
	const index_t length_y=10;
	SGMatrix<float64_t> m(length_x, length_y);
	CDenseMatrixOperator<float64_t> *op
			=new CDenseMatrixOperator<float64_t>(m);
	CDenseMatrixOperator<float64_t> *op_cloned
			=dynamic_cast<CDenseMatrixOperator<float64_t>*>(op->clone());
	SG_UNREF(op);
	SG_UNREF(op_cloned);
}
