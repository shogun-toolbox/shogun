/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Kunal Arora
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/LLTLinearSolver.h>

using namespace Eigen;

namespace shogun
{

CLLTLinearSolver::CLLTLinearSolver()
	: CLinearSolver<float64_t, float64_t>()
{
}

CLLTLinearSolver::~CLLTLinearSolver()
{
}

SGMatrix<float64_t> CLLTLinearSolver::compute_cholesky(
	CLinearOperator<float64_t>* A)
{
	//sanity check
	REQUIRE(A, "Operator is NULL!\n");
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(A);
	REQUIRE(op, "Operator is not SparseMatrixOperator type!\n");

	//creating eigen3 dense matrices
	SGMatrix<float64_t> mat_A=op->get_matrix_operator();
	Map<MatrixXd> map_A(mat_A.matrix, mat_A.num_rows, mat_A.num_cols);

	SGMatrix<float64_t> L(mat_A.num_rows,mat_A.num_cols);
	L.set_const(0.0);
	Map<MatrixXd> map_L(L.matrix, L.num_rows, L.num_cols);

	LLT<MatrixXd> llt(map_A);

	//compute matrix L
	map_L= llt.matrixL();

	// checking for success
	if (llt.info()==NumericalIssue)
		SG_WARNING("Matrix is not Hermitian positive definite!\n");

	//return matrix L
	return L;

}

SGVector<float64_t> CLLTLinearSolver::triangular_solve(
	SGMatrix<float64_t> L, SGVector<float64_t> b)
{
	//sanity check
	REQUIRE(L.num_cols==b.vlen, "Dimension mismatch!\n");

	// creating eigen3 maps for vectors
	SGVector<float64_t> x(L.num_cols);
	x.set_const(0.0);

	//creating eigen3 matrix
	Map<MatrixXd> map_L(L.matrix, L.num_rows, L.num_cols);

	//creating eigen3 vectors
	Map<VectorXd> map_x(x.vector, x.vlen);
	Map<VectorXd> map_b(b.vector, b.vlen);

	// x=L*^{-1}(L^{1}(b))
	map_x=map_L.triangularView<Eigen::Lower>().adjoint().solve(map_L.triangularView<Eigen::Lower>().solve(map_b));

	return x;

}
SGVector<float64_t> CLLTLinearSolver::solve(
		CLinearOperator<float64_t>* A, SGVector<float64_t> b)
{
	//compute the matrix L
	SGMatrix<float64_t> L=compute_cholesky(A);

	//solve the linear equation using triangular solve
	SGVector<float64_t>x =triangular_solve(L, b);

	return x;
}

}
