/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/eigsolver/DirectEigenSolver.h>

using namespace Eigen;

namespace shogun
{

DirectEigenSolver::DirectEigenSolver()
	: EigenSolver()
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

DirectEigenSolver::DirectEigenSolver(
	std::shared_ptr<DenseMatrixOperator<float64_t>> linear_operator)
	: EigenSolver(linear_operator->as<LinearOperator<float64_t>>())
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

DirectEigenSolver::~DirectEigenSolver()
{
	SG_TRACE("{} destroyed ({})", this->get_name(), fmt::ptr(this));
}

void DirectEigenSolver::compute()
{
	if (m_is_computed_min && m_is_computed_max)
	{
		SG_DEBUG("Minimum/maximum eigenvalues are already computed, exiting");
		return;
	}

	auto op
		=m_linear_operator->as<DenseMatrixOperator<float64_t>>();
	require(op, "Linear operator is not of CDenseMatrixOperator type!");

	SGMatrix<float64_t> m=op->get_matrix_operator();
	Map<MatrixXd> map_m(m.matrix, m.num_rows, m.num_cols);

	// compute the eigenvalues with Eigen3
	SelfAdjointEigenSolver<MatrixXd> eig_solver(map_m);
	m_min_eigenvalue=eig_solver.eigenvalues()[0];
	m_max_eigenvalue=eig_solver.eigenvalues()[op->get_dimension()-1];

	m_is_computed_min=true;
	m_is_computed_max=false;
}

}
