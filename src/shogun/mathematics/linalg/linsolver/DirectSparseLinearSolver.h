/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General turalPublic License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef DIRECT_SPARSE_LINEAR_SOLVER_H_
#define DIRECT_SPARSE_LINEAR_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** @brief Class that provides a solve method for real sparse-matrix
 * linear systems using LLT
 */
class CDirectSparseLinearSolver : public CLinearSolver<float64_t, float64_t>
{
public:
	/** default constructor */
	CDirectSparseLinearSolver();

	/** destructor */
	virtual ~CDirectSparseLinearSolver();

	/**
	 * solve method for solving real-valued sparse linear systems
	 *
	 * @param A the sparse linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A,
		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectSparseLinearSolver";
	}

};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_SPARSE_LINEAR_SOLVER_H_
