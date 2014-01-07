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
#include <mathematics/Math.h>
#include <mathematics/eigen3.h>
#include <mathematics/linalg/linop/SparseMatrixOperator.h>
#include <mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(ConjugateOrthogonalCGSolver, solve)
{
	const int32_t size=10;
	SGSparseMatrix<complex128_t> m(size, size);
	CSparseMatrixOperator<complex128_t>* A=new CSparseMatrixOperator<complex128_t>(m);

	// diagonal non-Hermintian matrix with random complex entries
	SGVector<complex128_t> diag(size);
	sg_rand->set_seed(100.0);
	for (index_t i=0; i<size; ++i)
	{
		float64_t real=sg_rand->std_normal_distrib();
		float64_t imag=sg_rand->std_normal_distrib();
		diag[i]=complex128_t(real, imag);
	}
	A->set_diagonal(diag);

	// vector b of the system
	SGVector<float64_t> b(size);
	for (index_t i=0; i<size; ++i)
		b[i]=sg_rand->std_normal_distrib();

	// Solve with COCG
	CConjugateOrthogonalCGSolver* cocg_linear_solver
		=new CConjugateOrthogonalCGSolver();
	const SGVector<complex128_t>& x=cocg_linear_solver->solve(A, b);

	const SGVector<complex128_t>& Ax=A->apply(x);

	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_Ax(Ax.vector, Ax.vlen);

	EXPECT_NEAR((map_b.cast<complex128_t>()-map_Ax).norm(), 0.0, 1E-10);

	SG_UNREF(A);
	SG_UNREF(cocg_linear_solver);
}
#endif //HAVE_EIGEN3
