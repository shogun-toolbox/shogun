/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General turalPublic License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Kunal Arora
 */

#ifndef LLT_LINEAR_SOLVER_H_
#define LLT_LINEAR_SOLVER_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** @brief Class that provides a solve method for real matrix
 * linear systems using LLT
 */
class CLLTLinearSolver : public CLinearSolver<float64_t, float64_t>
{
public:
	/** default constructor */
	CLLTLinearSolver();

	/** destructor */
	virtual ~CLLTLinearSolver();

	/**
	 * compute_cholesky method for computing the cholesky factor of linear systems
	 *
	 * @param A the linear operator of the system
	 * @return the cholesky factor
	 */
	virtual SGMatrix<float64_t> compute_cholesky(
		CLinearOperator<float64_t>* A);

	/**
	 * triangular_solve method for solving linear systems given their cholesky factor
	 *
	 * @param L the lower triangular matrix L i.e. the cholesky factor
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> triangular_solve(
		SGMatrix<float64_t> L, SGVector<float64_t> b);
	/**
	 * solve method for solving inear systems using LLT
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(CLinearOperator<float64_t>* A,
		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CLLTLinearSolver";
	}

};

}

#endif // DIRECT_SPARSE_LINEAR_SOLVER_H_
