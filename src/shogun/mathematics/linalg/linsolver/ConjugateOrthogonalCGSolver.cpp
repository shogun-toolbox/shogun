/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/common.h>


#include <shogun/lib/SGVector.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateOrthogonalCGSolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>
using namespace Eigen;

namespace shogun
{

ConjugateOrthogonalCGSolver::ConjugateOrthogonalCGSolver()
	: IterativeLinearSolver<complex128_t, float64_t>()
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

ConjugateOrthogonalCGSolver::ConjugateOrthogonalCGSolver(bool store_residuals)
	: IterativeLinearSolver<complex128_t, float64_t>(store_residuals)
{
	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

ConjugateOrthogonalCGSolver::~ConjugateOrthogonalCGSolver()
{
	SG_TRACE("{} destroyed ({})", this->get_name(), fmt::ptr(this));
}

SGVector<complex128_t> ConjugateOrthogonalCGSolver::solve(
	std::shared_ptr<LinearOperator<complex128_t>> A, SGVector<float64_t> b)
{
	SG_TRACE("ConjugateOrthogonalCGSolver::solve(): Entering..");

	// sanity check
	require(A, "Operator is NULL!");
	require(A->get_dimension()==b.vlen, "Dimension mismatch!\n, {} vs {}",
		A->get_dimension(), b.vlen);

	// the final solution vector, initial guess is 0
	SGVector<complex128_t> result(b.vlen);
	result.set_const(0.0);

	// the rest of the part hinges on eigen3 for computing norms
	Map<VectorXcd> x(result.vector, result.vlen);
	Map<VectorXd> b_map(b.vector, b.vlen);

	// direction vector
	SGVector<complex128_t> p_(result.vlen);
	Map<VectorXcd> p(p_.vector, p_.vlen);

	// residual r_i=b-Ax_i, here x_0=[0], so r_0=b
	VectorXcd r=b_map.cast<complex128_t>();

	// initial direction is same as residual
	p=r;

	// the iterator for this iterative solver
	IterativeSolverIterator<complex128_t> it(r, m_max_iteration_limit,
		m_relative_tolerence, m_absolute_tolerence);

	// start the timer
	Time time;
	time.start();

	// set the residuals to zero
	if (m_store_residuals)
		m_residuals.set_const(0.0);

	// CG iteration begins
	complex128_t r_norm2=r.transpose()*r;

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
		SGVector<complex128_t> Ap_=A->apply(p_);
		Map<VectorXcd> Ap(Ap_.vector, Ap_.vlen);

		// compute p^{T}Ap, if zero, failure
		complex128_t p_T_times_Ap=p.transpose()*Ap;
		if (p_T_times_Ap==0.0)
			break;

		// compute the alpha parameter of CG
		complex128_t alpha=r_norm2/p_T_times_Ap;

		// update the solution vector and residual
		// x_{i}=x_{i-1}+\alpha_{i}p
		x+=alpha*p;

		// r_{i}=r_{i-1}-\alpha_{i}p
		r-=alpha*Ap;

		// compute new ||r||_{2}, if zero, converged
		complex128_t r_norm2_i=r.transpose()*r;
		if (r_norm2_i==0.0)
			break;

		// compute the beta parameter of CG
		complex128_t beta=r_norm2_i/r_norm2;

		// update direction, and ||r||_{2}
		r_norm2=r_norm2_i;
		p=r+beta*p;
	}

	float64_t elapsed=time.cur_time_diff();

	if (!it.succeeded(r))
		io::warn("Did not converge!");

	io::info("Iteration took {} times, residual norm={:.20f}, time elapsed={}",
		it.get_iter_info().iteration_count, it.get_iter_info().residual_norm, elapsed);

	SG_DEBUG("ConjugateOrthogonalCGSolver::solve(): Leaving..");
	return result;
}

}
