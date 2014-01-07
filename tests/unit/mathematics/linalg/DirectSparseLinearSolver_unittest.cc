/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>

#ifdef HAVE_EIGEN3
#include <lib/SGVector.h>
#include <lib/SGSparseMatrix.h>
#include <mathematics/eigen3.h>
#include <mathematics/Math.h>
#include <mathematics/Random.h>
#include <mathematics/linalg/linop/SparseMatrixOperator.h>
#include <mathematics/linalg/linsolver/DirectSparseLinearSolver.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(DirectSparseLinearSolver, solve)
{
	const index_t size=100000;
	SGSparseMatrix<float64_t> sm(size, size);
	CSparseMatrixOperator<float64_t>* A=new CSparseMatrixOperator<float64_t>(sm);
	SGVector<float64_t> diag(size);
	float64_t difficulty=5;

	for (index_t i=0; i<size; ++i)
		diag[i]=CMath::pow(CMath::abs(sg_rand->std_normal_distrib()), difficulty)+0.0001;
	A->set_diagonal(diag);

	CDirectSparseLinearSolver* linear_solver=new CDirectSparseLinearSolver();
	SGVector<float64_t> b(size);
	b.set_const(0.5);

	const SGVector<float64_t>& x=linear_solver->solve(A, b);
	SGVector<float64_t> Ax=A->apply(x);
	Map<VectorXd> map_Ax(Ax.vector, Ax.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);

	EXPECT_NEAR((map_Ax-map_b).norm(), 0.0, 1E-10);

	SG_UNREF(linear_solver);
	SG_UNREF(A);
}
#endif // HAVE_EIGEN3
