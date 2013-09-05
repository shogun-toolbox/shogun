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
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/logdet/DenseMatrixOperator.h>
#include <shogun/mathematics/logdet/DirectEigenSolver.h>

using namespace Eigen;

namespace shogun
{

CDirectEigenSolver::CDirectEigenSolver()
	: CEigenSolver()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDirectEigenSolver::CDirectEigenSolver(
	CDenseMatrixOperator<float64_t>* linear_operator)
	: CEigenSolver((CLinearOperator<float64_t>*)linear_operator)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CDirectEigenSolver::~CDirectEigenSolver()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CDirectEigenSolver::compute()
{
	if (m_is_computed_min && m_is_computed_max)
	{
		SG_DEBUG("Minimum/maximum eigenvalues are already computed, exiting\n");
		return;
	}

	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);
	REQUIRE(op, "Linear operator is not of CDenseMatrixOperator type!\n");

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
#endif // HAVE_EIGEN3
