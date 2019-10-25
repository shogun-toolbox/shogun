/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Sunil Mahendrakar, Bjoern Esser
 */

#include <shogun/lib/common.h>


#include <shogun/lib/SGVector.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>

using namespace Eigen;

namespace shogun
{

CGMShiftedFamilySolver::CGMShiftedFamilySolver()
	: IterativeShiftedLinearFamilySolver<float64_t, complex128_t>()
{
}

CGMShiftedFamilySolver::CGMShiftedFamilySolver(bool store_residuals)
	: IterativeShiftedLinearFamilySolver<float64_t, complex128_t>(store_residuals)
{
}

CGMShiftedFamilySolver::~CGMShiftedFamilySolver()
{
}

SGVector<float64_t> CGMShiftedFamilySolver::solve(
	std::shared_ptr<LinearOperator<float64_t>> A, SGVector<float64_t> b)
{
	SGVector<complex128_t> shifts(1);
	shifts[0]=0.0;
	SGVector<complex128_t> weights(1);
	weights[0]=1.0;

	return solve_shifted_weighted(A, b, shifts, weights).get_real();
}

SGVector<complex128_t> CGMShiftedFamilySolver::solve_shifted_weighted(
	std::shared_ptr<LinearOperator<float64_t>> A, SGVector<float64_t> b,
	SGVector<complex128_t> shifts, SGVector<complex128_t> weights, bool negate)
{
	SG_TRACE("Entering");

	// sanity check
	require(A, "Operator is NULL!");
	require(A->get_dimension()==b.vlen, "Dimension mismatch! [{} vs {}]",
		A->get_dimension(), b.vlen);
	require(shifts.vector,"Shifts are not initialized!");
	require(weights.vector,"Weights are not initialized!");
	require(shifts.vlen==weights.vlen, "Number of shifts and number of "
		"weights are not equal! [{} vs {}]", shifts.vlen, weights.vlen);

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
	p_sh=r.replicate(1, shifts.vlen).cast<complex128_t>();

	// non shifted initializers
	float64_t r_norm2=r.dot(r);
	float64_t beta_old=1.0;
	float64_t alpha=1.0;

	// shifted quantities
	SGVector<complex128_t> alpha_sh(shifts.vlen);
	SGVector<complex128_t> beta_sh(shifts.vlen);
	SGVector<complex128_t> zeta_sh_old(shifts.vlen);
	SGVector<complex128_t> zeta_sh_cur(shifts.vlen);
	SGVector<complex128_t> zeta_sh_new(shifts.vlen);

	// shifted initializers
	zeta_sh_old.set_const(1.0);
	zeta_sh_cur.set_const(1.0);

	// the iterator for this iterative solver
	IterativeSolverIterator<float64_t> it(r, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// start the timer
	Time time;
	time.start();

	// set the residuals to zero
	if (m_store_residuals)
		m_residuals.set_const(0.0);

	// CG iteration begins
	for (it.begin(r); !it.end(r); ++it)
	{

		SG_DEBUG("CG iteration {}, residual norm {}",
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

		// compute the beta parameter of CG_M
		float64_t beta=-r_norm2/p_dot_Ap;

		// compute the zeta-shifted parameter of CG_M
		compute_zeta_sh_new(
			zeta_sh_old, zeta_sh_cur, shifts, beta_old, beta, alpha,
			zeta_sh_new, negate);

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

	float64_t elapsed=time.cur_time_diff();

	if (!it.succeeded(r))
		io::warn("Did not converge!");

	io::info("Iteration took {} times, residual norm={:.20f}, time elapsed={}",
		it.get_iter_info().iteration_count, it.get_iter_info().residual_norm, elapsed);

	// compute the final result vector multiplied by weights
	SGVector<complex128_t> result(b.vlen);
	result.set_const(0.0);
	Map<VectorXcd> x(result.vector, result.vlen);

	for (index_t i=0; i<x_sh.cols(); ++i)
		x+=x_sh.col(i)*weights[i];

	SG_TRACE("Leaving");
	return result;
}

}
