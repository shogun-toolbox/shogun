/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef LINEAR_SOLVER_H_
#define LINEAR_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
template<class T> class SGVector;
template<class RetType, class OperandType> class CLinearOperator;

/** @brief Abstract template base class that provides an abstract solve method
 * for linear systems, that takes a linear operator \f$A\f$, a vector \f$b\f$,
 * solves the system \f$Ax=b\f$ and returns the vector \f$x\f$.
 */
template<class T, class ST=T> class CLinearSolver : public CSGObject
{
public:
	/** default constructor */
	CLinearSolver()
	: CSGObject()
	{
	}

	/** destructor */
	virtual ~CLinearSolver()
	{
	}

	/**
	 * abstract solve method for solving linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<T> solve(CLinearOperator<SGVector<T>, SGVector<T> >* A, SGVector<ST> b) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LinearSolver";
	}

};
}

#endif // LINEAR_SOLVER_H_
