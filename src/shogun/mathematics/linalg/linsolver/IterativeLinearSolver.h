/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef ITERATIVE_LINEAR_SOLVER_H_
#define ITERATIVE_LINEAR_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** 
 * @brief abstract template base for all iterative linear solvers such as
 * conjugate gradient (CG) solvers. provides interface for setting the
 * iteration limit, relative/absolute tolerence. solve method is abstract.
 */
template<class T, class ST=T> class CIterativeLinearSolver : public CLinearSolver<T, ST>
{

public:
	/** default constructor */
	CIterativeLinearSolver();

	/** destructor */
	virtual ~CIterativeLinearSolver();

	/** 
	 * abstract solve method for solving real linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<T> solve(CLinearOperator<T>* A, SGVector<ST> b) = 0;

	/** set maximum iteration limit */
	void set_iteration_limit(int64_t iteration_limit)
	{
		m_max_iteration_limit=iteration_limit;
	}

	/** @return maximum iteration limit */
	const int64_t get_iteration_limit() const
	{
		return m_max_iteration_limit;
	}

	/** set relative tolerence */
	void set_relative_tolerence(float64_t relative_tolerence)
	{
		m_relative_tolerence=relative_tolerence;
	}

	/** @return relative tolerence */
	const float64_t get_relative_tolerence() const
	{
		return m_relative_tolerence;
	}

	/** set absolute tolerence */
	void set_absolute_tolerence(float64_t absolute_tolerence)
	{
		m_absolute_tolerence=absolute_tolerence;
	}

	/** @return absolute tolerence */
	const float64_t get_absolute_tolerence() const
	{
		return m_absolute_tolerence;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "IterativeLinearSolver";
	}

protected:

	/** iteration limit for conjugate gradient */
	int64_t m_max_iteration_limit;

	/** relative tolerence */
	float64_t m_relative_tolerence;

	/** absolute tolerence */
	float64_t m_absolute_tolerence;

private:
	/** initialize with default values and register params */
	void init();
};

}

#endif // ITERATIVE_LINEAR_SOLVER_H_
