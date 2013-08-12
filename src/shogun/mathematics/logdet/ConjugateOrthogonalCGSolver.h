/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef CONJUGATE_ORTHOGONAL_CG_SOLVER_H_
#define CONJUGATE_ORTHOGONAL_CG_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/logdet/IterativeLinearSolver.h>

namespace shogun
{
template<class T> class CLinearOperator;
template<class T> class SGVector;

/** 
 * @brief class that uses conjugate orthogonal conjugate gradient method of
 * solving a linear system involving a complex valued linear operator and
 * vector. Useful for large sparse systems involving sparse symmetric matrices
 * that are not Herimitian.
 *
 * Reference: Vorst, Melissen, "A Petrov-Galerkin Type Method for Solving Ax=b,
 * Where A Is Symmetric Complex". IEEE Transactions on Magnetics, Vol. 26,
 * No. 2, March 1990
 */
class CConjugateOrthogonalCGSolver
 : public CIterativeLinearSolver<complex64_t, float64_t>
{

public:
	/** default constructor */
	CConjugateOrthogonalCGSolver();

	/** destructor */
	virtual ~CConjugateOrthogonalCGSolver();

	/** 
	 * solve method for solving complex linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<complex64_t> solve(CLinearOperator<complex64_t>* A,

		SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "ConjugateOrthogonalCGSolver";
	}

};

}

#endif // HAVE_EIGEN3
#endif // CONJUGATE_ORTHOGONAL_CG_SOLVER_H_
