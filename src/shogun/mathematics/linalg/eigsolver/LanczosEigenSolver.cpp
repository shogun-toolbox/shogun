/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK

#include <shogun/base/Parameter.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>
#include <shogun/mathematics/linalg/eigsolver/LanczosEigenSolver.h>
#include <vector>
#include <shogun/lib/SGVector.h>

using namespace Eigen;

namespace shogun
{

CLanczosEigenSolver::CLanczosEigenSolver()
	: CEigenSolver()
{
	init();
}

CLanczosEigenSolver::CLanczosEigenSolver(
	CLinearOperator<float64_t>* linear_operator)
	: CEigenSolver(linear_operator)
{
	init();
}

void CLanczosEigenSolver::init()
{
	m_max_iteration_limit=1000;
	m_relative_tolerence=1E-6;
	m_absolute_tolerence=1E-6;

	SG_ADD(&m_max_iteration_limit, "max_iteration_limit",
		"Maximum number of iteration for the solver", MS_NOT_AVAILABLE);

	SG_ADD(&m_relative_tolerence, "relative_tolerence",
		"Relative tolerence of solver", MS_NOT_AVAILABLE);

	SG_ADD(&m_absolute_tolerence, "absolute_tolerence",
		"Absolute tolerence of solver", MS_NOT_AVAILABLE);
}

CLanczosEigenSolver::~CLanczosEigenSolver()
{
}

void CLanczosEigenSolver::compute()
{
	SG_DEBUG("Entering\n");

	if (m_is_computed_min && m_is_computed_max)
	{
		SG_DEBUG("Minimum/maximum eigenvalues are already computed, exiting\n");
		return;
	}

	REQUIRE(m_linear_operator, "Operator is NULL!\n");

	// vector v_0
	VectorXd v=VectorXd::Zero(m_linear_operator->get_dimension());

	// vector v_i, for i=1 this is random valued with norm 1
	SGVector<float64_t> v_(v.rows());
	Map<VectorXd> v_i(v_.vector, v_.vlen);
	v_i=VectorXd::Random(v_.vlen);
	v_i/=v_i.norm();

	// the iterator for this iterative solver
	IterativeSolverIterator<float64_t> it(v_i, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// the diagonal for Lanczos T-matrix (tridiagonal)
	std::vector<float64_t> alpha;

	// the subdiagonal for Lanczos T-matrix (tridiagonal)
	std::vector<float64_t> beta;

	float64_t beta_i=0.0;
	SGVector<float64_t> w_(v_.vlen);

	// CG iteration begins
	for (it.begin(v_i); !it.end(v_i); ++it)
	{
		SG_DEBUG("CG iteration %d, residual norm %f\n",
				it.get_iter_info().iteration_count,
				it.get_iter_info().residual_norm);

		// apply linear operator to the direction vector
		w_=m_linear_operator->apply(v_);
		Map<VectorXd> w_i(w_.vector, w_.vlen);

		// compute v^{T}Av, if zero, failure
		float64_t alpha_i=w_i.dot(v_i);
		if (alpha_i==0.0)
			break;

		// update w_i, v_(i-1) and find beta
		w_i-=alpha_i*v_i+beta_i*v;
		beta_i=w_i.norm();
		v=v_i;
		v_i=w_i/beta_i;

		// prepate Lanczos T-matrix from alpha and beta
		alpha.push_back(alpha_i);
		beta.push_back(beta_i);
	}

	// solve Lanczos T-matrix to get the eigenvalues
	int32_t M=0;
	SGVector<float64_t> w(alpha.size());
	int32_t info=0;

	// keeping copies of the diagonal and subdiagonal
	// because subsequent call to dstemr destroys it
	std::vector<float64_t> alpha_orig=alpha;
	std::vector<float64_t> beta_orig=beta;

	if (!m_is_computed_min)
	{
		// computing min eigenvalue
		wrap_dstemr('N', 'I', alpha.size(), &alpha[0], &beta[0],
			0.0, 0.0, 1, 1, &M, w.vector, NULL, 1, 1, NULL, 0.0, &info);

		if (info==0)
		{
			SG_INFO("Iteration took %ld times, residual norm=%.20lf\n",
			it.get_iter_info().iteration_count, it.get_iter_info().residual_norm);

			m_min_eigenvalue=w[0];
			m_is_computed_min=true;
		}
		else
			SG_WARNING("Some error occured while computing eigenvalues!\n");
	}

	if (!m_is_computed_max)
	{
		// computing max eigenvalue
		wrap_dstemr('N', 'I', alpha_orig.size(), &alpha_orig[0], &beta_orig[0],
			0.0, 0.0, w.vlen, w.vlen, &M, w.vector, NULL, 1, 1, NULL, 0.0, &info);

		if (info==0)
		{
			SG_INFO("Iteration took %ld times, residual norm=%.20lf\n",
			it.get_iter_info().iteration_count, it.get_iter_info().residual_norm);
			m_max_eigenvalue=w[0];
			m_is_computed_max=true;
		}
		else
			SG_WARNING("Some error occured while computing eigenvalues!\n");
	}

	SG_DEBUG("Leaving\n");
}

}
#endif // HAVE_LAPACK
