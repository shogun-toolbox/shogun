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
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>

using namespace Eigen;

namespace shogun
{

CCGMShiftedFamilySolver::CCGMShiftedFamilySolver()
	: CIterativeShiftedLinearFamilySolver<float64_t, complex64_t>()
{
}

CCGMShiftedFamilySolver::~CCGMShiftedFamilySolver()
{
}

SGVector<float64_t> CCGMShiftedFamilySolver::solve(
	CLinearOperator<float64_t>* A, SGVector<float64_t> b)
{
	SGVector<complex64_t> shifts(1);
	shifts[0]=0.0;
	SGVector<complex64_t> weights(1);
	weights[0]=1.0;

	return solve_shifted_weighted(A, b, shifts, weights).get_real();
}

SGVector<complex64_t> CCGMShiftedFamilySolver::solve_shifted_weighted(
	CLinearOperator<float64_t>* A, SGVector<float64_t> b,
	SGVector<complex64_t> shifts, SGVector<complex64_t> weights)
{
	SG_DEBUG("Entering\n");

	// sanity check
	REQUIRE(A, "Operator is NULL!\n");
	REQUIRE(A->get_dimension()==b.vlen, "Dimension mismatch! [%d vs %d]\n",
		A->get_dimension(), b.vlen);
	REQUIRE(shifts.vector,"Shifts are not initialized!\n");
	REQUIRE(weights.vector,"Weights are not initialized!\n");
	REQUIRE(shifts.vlen==weights.vlen, "Number of shifts and number of "
		"weights are not equal! [%d vs %d]\n", shifts.vlen, weights.vlen);

	// the solution matrix, one column per shift, initial guess 0 for all
	MatrixXcd x_sh=MatrixXcd::Zero(b.vlen, shifts.vlen);
	MatrixXcd p_sh=MatrixXcd::Zero(b.vlen, shifts.vlen);

	// non-shifted direction
	SGVector<float64_t> p_(b.vlen);

	// the rest of the part hinges on eigen3 for computing norms
	Map<VectorXd> b_map(b.vector, b.vlen);
	Map<VectorXd> p(p_.vector, p_.vlen);

	// residual r_i=b-Ax_i, here x_0=[0], so r_0=b
	VectorXd r=b_map;

	// initial direction is same as residual
	p=r;
	p_sh=r.replicate(1, shifts.vlen).cast<complex64_t>();

	// non shifted initializers
	float64_t r_norm2=r.dot(r);
	float64_t beta_old=1.0;
	float64_t alpha=1.0;

	// shifted quantities
	SGVector<complex64_t> alpha_sh(shifts.vlen);
	SGVector<complex64_t> beta_sh(shifts.vlen);
	SGVector<complex64_t> zeta_sh_old(shifts.vlen);
	SGVector<complex64_t> zeta_sh_cur(shifts.vlen);
	SGVector<complex64_t> zeta_sh_new(shifts.vlen);

	// shifted initializers
	zeta_sh_old.set_const(1.0);
	zeta_sh_cur.set_const(1.0);

	// the iterator for this iterative solver
	IterativeSolverIterator<float64_t> it(r, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// CG iteration begins
	for (it.begin(r); !it.end(r); ++it)
	{

		SG_DEBUG("CG iteration %d, residual norm %f\n",
				it.get_iter_info().iteration_count,
				it.get_iter_info().residual_norm);

		// apply linear operator to the direction vector
		SGVector<float64_t> Ap_=A->apply(p_);
		Map<VectorXd> Ap(Ap_.vector, Ap_.vlen);

		// compute p^{T}Ap, if zero, failure
		float64_t p_dot_Ap=p.dot(Ap);
		if (p_dot_Ap==0.0)
			break;

		// compute the beta parameter of CG_M
		float64_t beta=-r_norm2/p_dot_Ap;

		// compute the zeta-shifted parameter of CG_M
		compute_zeta_sh_new(zeta_sh_old, zeta_sh_cur, shifts, beta_old, beta,
			alpha, zeta_sh_new);

		// compute beta-shifted parameter of CG_M
		compute_beta_sh(zeta_sh_new, zeta_sh_cur, beta, beta_sh);

		// update the solution vector and residual
		for (index_t i=0; i<shifts.vlen; ++i)
			x_sh.col(i)-=beta_sh[i]*p_sh.col(i);

		// r_{i}=r_{i-1}+\beta_{i}Ap
		r+=beta*Ap;

		// compute new ||r||_{2}, if zero, converged
		float64_t r_norm2_i=r.dot(r);
		if (r_norm2_i==0.0)
			break;

		// compute the alpha parameter of CG_M
		alpha=r_norm2_i/r_norm2;

		// update ||r||_{2}
		r_norm2=r_norm2_i;

		// update direction
		p=r+alpha*p;

		compute_alpha_sh(zeta_sh_new, zeta_sh_cur, beta_sh, beta, alpha, alpha_sh);

		for (index_t i=0; i<shifts.vlen; ++i)
		{		
			p_sh.col(i)*=alpha_sh[i];
			p_sh.col(i)+=zeta_sh_new[i]*r;
		}

		// update parameters
		for (index_t i=0; i<shifts.vlen; ++i)
		{
			zeta_sh_old[i]=zeta_sh_cur[i];
			zeta_sh_cur[i]=zeta_sh_new[i];
		}
		beta_old=beta;
	}

	if (it.succeeded(r))
	{
		SG_DEBUG("Iteration took %ld times, residual norm=%.20lf\n",
			it.get_iter_info().iteration_count, it.get_iter_info().residual_norm);
	}
	else
		SG_WARNING("Did not converge!\n");

	// compute the final result vector multiplied by weights
	SGVector<complex64_t> result(b.vlen);
	result.set_const(0.0);
	Map<VectorXcd> x(result.vector, result.vlen);

	for (index_t i=0; i<x_sh.cols(); ++i)
		x+=x_sh.col(i)*weights[i];

	SG_DEBUG("Leaving\n");
	return result;
}

}
#endif // HAVE_EIGEN3
