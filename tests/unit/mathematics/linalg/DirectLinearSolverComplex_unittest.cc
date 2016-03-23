/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(DirectLinearSolverComplex, solve_SVD)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	CDenseMatrixOperator<complex128_t>* A
		=new CDenseMatrixOperator<complex128_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CDirectLinearSolverComplex solver(DS_SVD);
	SGVector<complex128_t> x
		=solver.solve((CLinearOperator<complex128_t>*)A, b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);

	SG_UNREF(A);
}

TEST(DirectLinearSolverComplex, solve_QR_NOPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	CDenseMatrixOperator<complex128_t>* A
		=new CDenseMatrixOperator<complex128_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CDirectLinearSolverComplex solver(DS_QR_NOPERM);
	SGVector<complex128_t> x
		=solver.solve((CLinearOperator<complex128_t>*)A, b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);

	SG_UNREF(A);
}

TEST(DirectLinearSolverComplex, solve_QR_COLPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	CDenseMatrixOperator<complex128_t>* A
		=new CDenseMatrixOperator<complex128_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CDirectLinearSolverComplex solver(DS_QR_COLPERM);
	SGVector<complex128_t> x
		=solver.solve((CLinearOperator<complex128_t>*)A, b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);

	SG_UNREF(A);
}

TEST(DirectLinearSolverComplex, solve_QR_FULLPERM)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	m(0,0)=complex128_t(2.0);
	m(0,1)=complex128_t(1.0, 2.0);
	m(1,0)=complex128_t(1.0, 2.0);
	m(1,1)=complex128_t(3.0);

	CDenseMatrixOperator<complex128_t>* A
		=new CDenseMatrixOperator<complex128_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CDirectLinearSolverComplex solver(DS_QR_FULLPERM);
	SGVector<complex128_t> x
		=solver.solve((CLinearOperator<complex128_t>*)A, b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);

	SG_UNREF(A);
}

TEST(DirectLinearSolverComplex, solve_LLT)
{
	const index_t size=2;
	SGMatrix<complex128_t> m(size, size);
	// LLT doesn't work on non-symmetric matrices
	m(0,0)=complex128_t(2.0, 0.0);
	m(0,1)=complex128_t(1.0, 0.0);
	m(1,0)=complex128_t(1.0, 0.0);
	m(1,1)=complex128_t(2.5, 0.0);

	CDenseMatrixOperator<complex128_t>* A
		=new CDenseMatrixOperator<complex128_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CDirectLinearSolverComplex solver(DS_LLT);
	SGVector<complex128_t> x
		=solver.solve((CLinearOperator<complex128_t>*)A, b);

	SGVector<complex128_t> bp=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_bp(bp.vector, bp.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_bp).norm()/map_b.norm(),
		0.0, 1E-15);

	SG_UNREF(A);
}
