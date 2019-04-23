/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectSparseLinearSolver.h>

using namespace Eigen;

namespace shogun
{

DirectSparseLinearSolver::DirectSparseLinearSolver()
	: LinearSolver<float64_t, float64_t>()
{
}

DirectSparseLinearSolver::~DirectSparseLinearSolver()
{
}

SGVector<float64_t> DirectSparseLinearSolver::solve(
		std::shared_ptr<LinearOperator<float64_t>> A, SGVector<float64_t> b)
{
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch!\n");
	auto op=A->as<SparseMatrixOperator<float64_t>>();
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
