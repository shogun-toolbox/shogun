/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef CG_M_SHIFTED_FAMILY_SOLVER_H_
#define CG_M_SHIFTED_FAMILY_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/logdet/IterativeShiftedLinearFamilySolver.h>

namespace shogun
{
template<class T> class CLinearOperator;
template<class T> class SGVector;

/** 
 * @brief class that uses conjugate gradient method for solving a shifted
 * linear system family where the linear opeator is real valued and symmetric
 * positive definite, the vector is real valued, but the shifts are complex
 *
 * Note: The implementation of solve_shifted_weighted has been adapted from the
 * open source library Krylstat (https://github.com/Froskekongen/KRYLSTAT/),
 * written by Erlend Aune, under GPL2+
 */
class CCGMShiftedFamilySolver
 : public CIterativeShiftedLinearFamilySolver<float64_t, complex64_t>
{

public:
	/** default constructor */
	CCGMShiftedFamilySolver();

	/** destructor */
	virtual ~CCGMShiftedFamilySolver();

	/** 
	 * solve method for solving linear systems assuming no shift
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(CLinearOperator<float64_t>* A,
		SGVector<float64_t> b);

	/**
	 * method that solves the shifted family of linear systems, multiples
	 * each solution vector with a weight, computes a summation over all the
	 * shifts and returns the final solution vector
	 * 
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @param shifts the shifts of the shifted system
	 * @param weights the weights to be multiplied with each solution for each
	 * shift
	 */
	virtual SGVector<complex64_t> solve_shifted_weighted(
		CLinearOperator<float64_t>* A, SGVector<float64_t> b,
		SGVector<complex64_t> shifts, SGVector<complex64_t> weights);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CGMShiftedFamilySolver";
	}

};

}

#endif // HAVE_EIGEN3
#endif // CG_M_SHIFTED_FAMILY_SOLVER_H_
