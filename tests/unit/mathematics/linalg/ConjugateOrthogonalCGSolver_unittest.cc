/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
 
#include <shogun/lib/common.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <shogun/features/SparseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(ConjugateOrthogonalCGSolver, solve)
{
	const int32_t size=2;
	SGMatrix<complex64_t> m(size, size);
	m.set_const(0.0);

	// diagonal non-Hermintian matrix
	for (index_t i=0; i<size; ++i)
		m(i,i)=complex64_t(i+1, i+1);

	// constant vector of the system
	SGVector<float64_t> b(size);
	b.set_const(0.5);
	
	// Creating sparse system to solve with COCG
	CSparseFeatures<complex64_t> feat(m);
	SGSparseMatrix<complex64_t> mat=feat.get_sparse_feature_matrix();
	CSparseMatrixOperator<complex64_t>* A
		=new CSparseMatrixOperator<complex64_t>(mat);

	// Solve with COCG
	CConjugateOrthogonalCGSolver cocg_linear_solver;
	cocg_linear_solver.set_iteration_limit(5000);
	SGVector<complex64_t> x_cg=cocg_linear_solver.solve(A, b);

	// Creating dense system to solve with direct solver
	CDenseMatrixOperator<complex64_t>* B
		=new CDenseMatrixOperator<complex64_t>(m);

	// Solve with direct triangular solver
	CDirectLinearSolverComplex direct_linear_solver;
	SGVector<complex64_t> x_direct=direct_linear_solver.solve(B, b);

	Map<VectorXcd> map_x_cg(x_cg.vector, x_cg.vlen);
	Map<VectorXcd> map_x_direct(x_direct.vector, x_direct.vlen);

	EXPECT_NEAR((map_x_cg-map_x_direct).norm(), 0.0, 0.1);

	SG_UNREF(A);
	SG_UNREF(B);
}
#endif //HAVE_EIGEN3
