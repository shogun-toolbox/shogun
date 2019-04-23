/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/DirectLinearSolverComplex.h>

using namespace Eigen;

namespace shogun
{

DirectLinearSolverComplex::DirectLinearSolverComplex()
	: LinearSolver<complex128_t, float64_t>(),
	  m_type(DS_QR_NOPERM)
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

DirectLinearSolverComplex::DirectLinearSolverComplex(EDirectSolverType type)
	: LinearSolver<complex128_t, float64_t>(),
	  m_type(type)
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

DirectLinearSolverComplex::~DirectLinearSolverComplex()
{
	SG_TRACE("{} destroyed ({})", this->get_name(), fmt::ptr(this));
}

SGVector<complex128_t> DirectLinearSolverComplex::solve(
		std::shared_ptr<LinearOperator<complex128_t>> A, SGVector<float64_t> b)
{
	SGVector<complex128_t> x(b.vlen);

	require(A, "Operator is NULL!");
	require(A->get_dimension()==b.vlen, "Dimension mismatch!");

	auto op=A->as<DenseMatrixOperator<complex128_t>>();
	require(op, "Operator is not DenseMatrixOperator<complex128_t, float64_t> type!");

	SGMatrix<complex128_t> mat_A=op->get_matrix_operator();
	Map<MatrixXcd> map_A(mat_A.matrix, mat_A.num_rows, mat_A.num_cols);
	Map<VectorXd> map_b(b.vector, b.vlen);
	Map<VectorXcd> map_x(x.vector, x.vlen);

	switch (m_type)
	{
	case DS_LLT:
		{
			LLT<MatrixXcd> llt(map_A);
			map_x=llt.solve(map_b.cast<complex128_t>());

			// checking for success
			if (llt.info()==NumericalIssue)
				io::warn("Matrix is not Hermitian positive definite!");
		}
		break;
	case DS_QR_NOPERM:
		map_x=map_A.householderQr().solve(map_b.cast<complex128_t>());
		break;
	case DS_QR_COLPERM:
		map_x=map_A.colPivHouseholderQr().solve(map_b.cast<complex128_t>());
		break;
	case DS_QR_FULLPERM:
		map_x=map_A.fullPivHouseholderQr().solve(map_b.cast<complex128_t>());
		break;
	case DS_SVD:
		map_x=map_A.jacobiSvd(ComputeThinU|ComputeThinV).solve(map_b.cast<complex128_t>());
		break;
	}

	return x;
}

}
