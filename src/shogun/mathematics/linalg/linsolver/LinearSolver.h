/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Bjoern Esser
 */

#ifndef LINEAR_SOLVER_H_
#define LINEAR_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class LinearOperator;

/** @brief Abstract template base class that provides an abstract solve method
 * for linear systems, that takes a linear operator \f$A\f$, a vector \f$b\f$,
 * solves the system \f$Ax=b\f$ and returns the vector \f$x\f$.
 */
template<class T, class ST=T> class LinearSolver : public SGObject
{
public:
	/** default constructor */
	LinearSolver()
	: SGObject()
	{
	}

	/** destructor */
	virtual ~LinearSolver()
	{
	}

	/**
	 * abstract solve method for solving linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<T> solve(std::shared_ptr<LinearOperator<T>> A, SGVector<ST> b) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LinearSolver";
	}

};
}

#endif // LINEAR_SOLVER_H_
