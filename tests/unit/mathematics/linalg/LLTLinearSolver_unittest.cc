/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Kunal Arora
 */

#include <shogun/lib/common.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/MatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/LLTLinearSolver.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(LLTLinearSolver, solve)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	// LLT doesn't work on non-symmetric matrices
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	CDenseMatrixOperator<float64_t>* A
		=new CDenseMatrixOperator<float64_t>(m);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	CLLTLinearSolver* linear_solver=new CLLTLinearSolver();
	SGVector<float64_t> x
		=linear_solver->solve((CLinearOperator<float64_t>*)A, b);

	SGVector<float64_t> Ax=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXd> map_Ax(Ax.vector, Ax.vlen);

	EXPECT_NEAR((map_Ax-map_b).norm(),
		0.0, 1E-15);

	SG_UNREF(linear_solver);
	SG_UNREF(A);
}

TEST(LLTLinearSolver, compute_cholesky)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	// LLT doesn't work on non-symmetric matrices
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	CDenseMatrixOperator<float64_t>* A
		=new CDenseMatrixOperator<float64_t>(m);

	CLLTLinearSolver* linear_solver=new CLLTLinearSolver();
	SGMatrix<float64_t> L
		=linear_solver->compute_cholesky((CLinearOperator<float64_t>*)A);


	Map<MatrixXd> map_A(m.matrix,m.num_rows,m.num_cols);
	Map<MatrixXd> map_L(L.matrix,L.num_rows,L.num_cols);
	EXPECT_NEAR((map_A-map_L*map_L.transpose()).norm(),
		0.0, 1E-15);

	SG_UNREF(linear_solver);
	SG_UNREF(A);
}

TEST(LLTLinearSolver, triangular_solve)
{
	const index_t size=2;
	SGMatrix<float64_t> m(size, size);

	// LLT doesn't work on non-symmetric matrices
	m(0,0)=2.0;
	m(0,1)=1.0;
	m(1,0)=1.0;
	m(1,1)=2.5;

	CDenseMatrixOperator<float64_t>* A
		=new CDenseMatrixOperator<float64_t>(m);

	CLLTLinearSolver* linear_solver=new CLLTLinearSolver();
	SGMatrix<float64_t> L
		=linear_solver->compute_cholesky((CLinearOperator<float64_t>*)A);

	SGVector<float64_t> b(size);
	b.set_const(1.0);

	SGVector<float64_t> x
		=linear_solver->triangular_solve(L, b);

	SGVector<float64_t> Ax=A->apply(x);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXd> map_Ax(Ax.vector, Ax.vlen);

	EXPECT_NEAR((map_Ax-map_b).norm(),
		0.0, 1E-15);

	SG_UNREF(linear_solver);
	SG_UNREF(A);
}

