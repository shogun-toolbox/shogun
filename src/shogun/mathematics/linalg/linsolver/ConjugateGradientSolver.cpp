/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/common.h>


#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/ConjugateGradientSolver.h>
#include <shogun/mathematics/linalg/linsolver/IterativeSolverIterator.h>

using namespace Eigen;

namespace shogun
{

ConjugateGradientSolver::ConjugateGradientSolver()
	: IterativeLinearSolver<float64_t>()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this);
}

ConjugateGradientSolver::ConjugateGradientSolver(bool store_residuals)
	: IterativeLinearSolver<float64_t>(store_residuals)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this);
}

ConjugateGradientSolver::~ConjugateGradientSolver()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this);
}

SGVector<float64_t> ConjugateGradientSolver::solve(
	std::shared_ptr<LinearOperator<float64_t>> A, SGVector<float64_t> b)
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
	Time time;
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
