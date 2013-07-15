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
template<class T, class ST> class CLinearOperator;

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
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CLinearSolver()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** 
	 * abstract solve method for solving linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<T> solve(CLinearOperator<T, ST>* A, SGVector<ST> b) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LinearSolver";
	}

};

template class CLinearSolver<bool>;
template class CLinearSolver<char>;
template class CLinearSolver<int8_t>;
template class CLinearSolver<uint8_t>;
template class CLinearSolver<int16_t>;
template class CLinearSolver<uint16_t>;
template class CLinearSolver<int32_t>;
template class CLinearSolver<uint32_t>;
template class CLinearSolver<int64_t>;
template class CLinearSolver<uint64_t>;
template class CLinearSolver<float32_t>;
template class CLinearSolver<float64_t>;
template class CLinearSolver<floatmax_t>;
template class CLinearSolver<complex64_t>;
template class CLinearSolver<complex64_t, float64_t>;

}

#endif // LINEAR_SOLVER_H_
