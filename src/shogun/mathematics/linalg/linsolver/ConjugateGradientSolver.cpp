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
#include <shogun/lib/Time.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>

using namespace Eigen;

namespace shogun
{

CConjugateGradientSolver::CConjugateGradientSolver()
	: CIterativeLinearSolver<float64_t>()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this);
}

CConjugateGradientSolver::CConjugateGradientSolver(bool store_residuals)
	: CIterativeLinearSolver<float64_t>(store_residuals)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this);
}

CConjugateGradientSolver::~CConjugateGradientSolver()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this);
}

SGVector<float64_t> CConjugateGradientSolver::solve(
	CLinearOperator<float64_t>* A, SGVector<float64_t> b)
{
	SG_DEBUG("CConjugateGradientSolve::solve(): Entering..\n");

	// sanity check
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch!\n");

	// the final solution vector, initial guess is 0
	SGVector<float64_t> result(b.vlen);
	result.set_const(0.0);

	// the rest of the part hinges on eigen3 for computing norms
	Map<VectorXd> x(result.vector, result.vlen);
	Map<VectorXd> b_map(b.vector, b.vlen);

	// direction vector
	SGVector<float64_t> p_(result.vlen);
	Map<VectorXd> p(p_.vector, p_.vlen);

	// residual r_i=b-Ax_i, here x_0=[0], so r_0=b
	VectorXd r=b_map;

	// initial direction is same as residual
	p=r;

	// the iterator for this iterative solver
	IterativeSolverIterator<float64_t> it(b_map, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// CG iteration begins
	float64_t r_norm2=r.dot(r);

	// start the timer
	CTime time;
	time.start();

	// set the residuals to zero
	if (m_store_residuals)
		m_residuals.set_const(0.0);

	for (it.begin(r); !it.end(r); ++it)
	{
		SG_DEBUG("CG iteration %d, residual norm %f\n",
			it.get_iter_info().iteration_count,
			it.get_iter_info().residual_norm);

		if (m_store_residuals)
		{
			m_residuals[it.get_iter_info().iteration_count]
				=it.get_iter_info().residual_norm;
		}

		// apply linear operator to the direction vector
		SGVector<float64_t> Ap_=A->apply(p_);
		Map<VectorXd> Ap(Ap_.vector, Ap_.vlen);

		// compute p^{T}Ap, if zero, failure
		float64_t p_dot_Ap=p.dot(Ap);
		if (p_dot_Ap==0.0)
			break;

		// compute the alpha parameter of CG
		float64_t alpha=r_norm2/p_dot_Ap;

		// update the solution vector and residual
		// x_{i}=x_{i-1}+\alpha_{i}p
		x+=alpha*p;

		// r_{i}=r_{i-1}-\alpha_{i}p
		r-=alpha*Ap;

		// compute new ||r||_{2}, if zero, converged
		float64_t r_norm2_i=r.dot(r);
		if (r_norm2_i==0.0)
			break;

		// compute the beta parameter of CG
		float64_t beta=r_norm2_i/r_norm2;

		// update direction, and ||r||_{2}
		r_norm2=r_norm2_i;
		p=r+beta*p;
	}

	float64_t elapsed=time.cur_time_diff();

	if (!it.succeeded(r))
		SG_WARNING("Did not converge!\n");

	SG_INFO("Iteration took %ld times, residual norm=%.20lf, time elapsed=%lf\n",
		it.get_iter_info().iteration_count, it.get_iter_info().residual_norm, elapsed);

	SG_DEBUG("CConjugateGradientSolve::solve(): Leaving..\n");
	return result;
}

}
#endif // HAVE_EIGEN3
