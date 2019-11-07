/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#ifndef DIRECT_SPARSE_LINEAR_SOLVER_H_
#define DIRECT_SPARSE_LINEAR_SOLVER_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** @brief Class that provides a solve method for real sparse-matrix
 * linear systems using LLT
 */
class DirectSparseLinearSolver : public LinearSolver<float64_t, float64_t>
{
public:
	/** default constructor */
	DirectSparseLinearSolver();

	/** destructor */
	virtual ~DirectSparseLinearSolver();

	/**
	 * solve method for solving real-valued sparse linear systems
	 *
	 * @param A the sparse linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(std::shared_ptr<LinearOperator<float64_t>> A,
		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectSparseLinearSolver";
	}

};

}

#endif // DIRECT_SPARSE_LINEAR_SOLVER_H_
