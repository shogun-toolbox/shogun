/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef ITERATIVE_SOLVER_ITERATOR_H_
#define ITERATIVE_SOLVER_ITERATOR_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

/**
 * @brief struct that contains current state of the iteration for iterative
 * linear solvers
 */
typedef struct _IterInfo
{
	/** norm of current residual */
	float64_t residual_norm;

	/** iteration count */
	index_t iteration_count;
} IterInfo;

/**
 * @brief template class that is used as an iterator for an iterative
 * linear solver. In the iteration of solving phase, each solver initializes
 * the iteration with a maximum number of iteration limit, and relative/
 * absolute tolerence. They then call begin with the residual vector and
 * continue until its end returns true, i.e. either it has converged or
 * iteration count reached maximum limit.
 */
template<class T> class IterativeSolverIterator
{

typedef Matrix<T, Dynamic, 1> VectorXt;

public:
	/**
	 * constructor
	 *
	 * tolerence of the solver is absolute_tolerence + relative_tolerence * ||b||
	 *
	 * @param b the vector of the linear system Ax=b
	 * @param max_iteration_limit maximum iteration limit
	 * @param relative_tolerence relative tolerence of the iterative method
	 * @param absolute_tolerence absolute tolerence of the iterative method
	 */
	IterativeSolverIterator(const VectorXt& b,
		index_t max_iteration_limit=1000,
		float64_t relative_tolerence=1E-5,
		float64_t absolute_tolerence=1E-5)
	: m_max_iteration_limit(max_iteration_limit),
		m_tolerence(absolute_tolerence+relative_tolerence*b.norm()),
		m_success(false)
	{
		m_iter_info.residual_norm=std::numeric_limits<float64_t>::max();
		m_iter_info.iteration_count=0;
	}

	/** assign operator from an IterInfo */
	void begin(const VectorXt& residual)
	{
		m_iter_info.residual_norm=residual.norm();
		m_iter_info.iteration_count=0;
	}

	/** @return true if converged or maximum iteration limit crossed */
	const bool end(const VectorXt& residual)
	{
		m_iter_info.residual_norm=residual.norm();

		m_success=m_iter_info.residual_norm < m_tolerence;
		return m_success || m_iter_info.iteration_count >= m_max_iteration_limit;
	}

	/** @return current iteration info */
	const IterInfo get_iter_info() const
	{
		return m_iter_info;
	}

	/** @return success status */
	const bool succeeded(const VectorXt& residual)
	{
		m_iter_info.residual_norm=residual.norm();

		m_success=m_iter_info.residual_norm < m_tolerence;
		return m_success;
	}

	/** increment operator */
	void operator++()
	{
		m_iter_info.iteration_count++;
	}

private:
	/** iteration info */
	IterInfo m_iter_info;

	/** maximum iteration limit */
	const index_t m_max_iteration_limit;

	/** tolerence of the iterative solver */
	const float64_t m_tolerence;

	/** true if converged successfully, false otherwise */
	bool m_success;
};

}

#endif // HAVE_EIGEN3
#endif // ITERATIVE_SOLVER_ITERATOR_H_
