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
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/logdet/LinearOperator.h>
#include <shogun/mathematics/logdet/ConjugateOrthogonalCGSolver.h>
#include <shogun/mathematics/logdet/IterativeSolverIterator.h>
using namespace Eigen;

namespace shogun
{

CConjugateOrthogonalCGSolver::CConjugateOrthogonalCGSolver()
	: CIterativeLinearSolver<complex64_t, float64_t>()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this);
}

CConjugateOrthogonalCGSolver::~CConjugateOrthogonalCGSolver()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this);
}

SGVector<complex64_t> CConjugateOrthogonalCGSolver::solve(
	CLinearOperator<complex64_t>* A, SGVector<float64_t> b)
{
	SG_DEBUG("CConjugateOrthogonalCGSolver::solve(): Entering..\n");

	// sanity check
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch!\n, %d vs %d",
		A->get_dimension(), b.vlen);

	// the final solution vector, initial guess is 0
	SGVector<complex64_t> result(b.vlen);
	result.set_const(0.0);

	// the rest of the part hinges on eigen3 for computing norms
	Map<VectorXcd> x(result.vector, result.vlen);
	Map<VectorXd> b_map(b.vector, b.vlen);

	// direction vector
	SGVector<complex64_t> p_(result.vlen);
	Map<VectorXcd> p(p_.vector, p_.vlen);

	// residual r_i=b-Ax_i, here x_0=[0], so r_0=b
	VectorXcd r=b_map.cast<complex64_t>();

	// initial direction is same as residual
	p=r;

	// the iterator for this iterative solver
	IterativeSolverIterator<complex64_t> it(r, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// CG iteration begins
	float64_t r_norm2=(r.dot(r)).real();

	for (it.begin(r); !it.end(r); ++it)
	{
		// apply linear operator to the direction vector
		SGVector<complex64_t> Ap_=A->apply(p_);
		Map<VectorXcd> Ap(Ap_.vector, Ap_.vlen);

		// compute p^{T}Ap, if zero, failure
		complex64_t p_T_times_Ap=p.transpose()*Ap;
		if (p_T_times_Ap==0.0)
			break;

		// compute the alpha parameter of CG
		complex64_t alpha=r_norm2/p_T_times_Ap;

		// update the solution vector and residual
		// x_{i}=x_{i-1}+\alpha_{i}p
		x+=alpha*p;

		// r_{i}=r_{i-1}-\alpha_{i}p
		r-=alpha*Ap;

		// compute new ||r||_{2}, if zero, converged
		float64_t r_norm2_i=(r.dot(r)).real();
		if (r_norm2_i==0.0)
			break;

		// compute the beta parameter of CG
		float64_t beta=r_norm2_i/r_norm2;

		// update direction, and ||r||_{2}
		r_norm2=r_norm2_i;
		p=r+beta*p;
	}

	if (it.succeeded(r))
	{
		SG_DEBUG("Iteration took %ld times, residual norm=%.20lf\n",
			it.get_iter_info().iteration_count, it.get_iter_info().residual_norm);
	}
	else
		SG_WARNING("Did not converge!\n");

	SG_DEBUG("CConjugateOrthogonalCGSolver::solve(): Leaving..\n");
	return result;
}

}
#endif // HAVE_EIGEN3
