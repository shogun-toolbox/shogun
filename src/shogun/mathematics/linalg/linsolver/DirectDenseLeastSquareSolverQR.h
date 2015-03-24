/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General turalPublic License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2015 Yingrui Chang
 */

#ifndef DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H_
#define DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** @brief Class that provides a solve method for least square problem
 * using QR decomposition
 */
class CDirectDenseLeastSquareSolverQR : public CLinearSolver<float64_t, float64_t>
{
public:
	/** default constructor */
	CDirectDenseLeastSquareSolverQR();

	/** destructor */
	virtual ~CDirectDenseLeastSquareSolverQR();

	/**
	 * solve method for solving real-valued least square problem based on QR decomposition
	 *
	 * @param A the dense linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(CLinearOperator<SGVector<float64_t>, SGVector<float64_t> >* A,
		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectDenseLeastSquareSolverQR";
	}

};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_DENSE_LEAST_SQUARE_SOLVER_QR_H_
