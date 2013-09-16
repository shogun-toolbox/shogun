/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectSparseLinearSolver.h>

using namespace Eigen;

namespace shogun
{

CDirectSparseLinearSolver::CDirectSparseLinearSolver()
	: CLinearSolver<float64_t, float64_t>()
{
}

CDirectSparseLinearSolver::~CDirectSparseLinearSolver()
{
}

SGVector<float64_t> CDirectSparseLinearSolver::solve(
		CLinearOperator<float64_t>* A, SGVector<float64_t> b)
{
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch!\n");
	CSparseMatrixOperator<float64_t>* op
		=dynamic_cast<CSparseMatrixOperator<float64_t>*>(A);
	REQUIRE(op, "Operator is not SparseMatrixOperator type!\n");

	// creating eigen3 Sparse Matrix
	SGSparseMatrix<float64_t> sm=op->get_matrix_operator();
	typedef SparseMatrix<float64_t> MatrixType;
	const MatrixType& m=EigenSparseUtil<float64_t>::toEigenSparse(sm);

	// creating eigen3 maps for vectors
	SGVector<float64_t> x(m.cols());
	x.set_const(0.0);
	Map<VectorXd> map_x(x.vector, x.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);

	// using LLT to solve the system Ax=b
	SimplicialLLT<MatrixType> llt;
	llt.compute(m);
	map_x=llt.solve(map_b);

	// checking for success
	if (llt.info()==NumericalIssue)
		SG_WARNING("Matrix is not Hermitian positive definite!\n");

	return x;
}

}
#endif // HAVE_EIGEN3
