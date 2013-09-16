/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef CONJUGATE_GRADIENT_SOLVER_H_
#define CONJUGATE_GRADIENT_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>

namespace shogun
{
template<class T> class CLinearOperator;
template<class T> class SGVector;

/** 
 * @brief class that uses conjugate gradient method of solving a linear system
 * involving a real valued linear operator and vector. Useful for large sparse
 * systems involving sparse symmetric and positive-definite matrices.
 */
class CConjugateGradientSolver : public CIterativeLinearSolver<float64_t, float64_t>
{

public:
	/** default constructor */
	CConjugateGradientSolver();

	/** destructor */
	virtual ~CConjugateGradientSolver();

	/** 
	 * solve method for solving real linear systems
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
		return "ConjugateGradientSolver";
	}

};

}

#endif // HAVE_EIGEN3
#endif // CONJUGATE_GRADIENT_SOLVER_H_
