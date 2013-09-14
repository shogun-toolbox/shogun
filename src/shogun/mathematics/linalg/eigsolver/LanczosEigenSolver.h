/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef LANCZOS_EIGEN_SOLVER_H_
#define LANCZOS_EIGEN_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3
#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>

namespace shogun
{
template<class T> class CLinearOperator;

/** @brief Class that computes eigenvalues of a real valued, self-adjoint
 * linear operator using Lanczos algorithm
 */
class CLanczosEigenSolver : public CEigenSolver
{
public:
	/** default constructor */
	CLanczosEigenSolver();

	/** 
	 * constructor
	 * 
	 * @param linear_operator self-adjoint linear operator whose eigenvalues 
	 * are to be found
	 */
	CLanczosEigenSolver(CLinearOperator<float64_t>* linear_operator);

	/** destructor */
	virtual ~CLanczosEigenSolver();

	/** 
	 * compute method for computing eigenvalues of a real valued linear operator
	 */
	virtual void compute();

	/** @param max_iteration_limit to be set */
	void set_max_iteration_limit(int64_t max_iteration_limit)
	{
		m_max_iteration_limit=max_iteration_limit;
	}

	/** @return max iteration limit */
	const int64_t get_max_iteration_limit() const
	{
		return m_max_iteration_limit;
	}

	/** @param relative_tolerence to be set */
	void set_relative_tolerence(float64_t relative_tolerence)
	{
		m_relative_tolerence=relative_tolerence;
	}

	/** @return relative tolerence */
	const float64_t get_relative_tolerence() const
	{
		return m_relative_tolerence;
	}

	/** @param absolute_tolerence to be set */
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
		return "LanczosEigenSolver";
	}

private:
	/** maximum iteration limit */
	int64_t m_max_iteration_limit;

	/** relative tolerence */
	float64_t m_relative_tolerence;

	/** absolute tolerence */
	float64_t m_absolute_tolerence;

	/** register params and initialize with default values */
	void init();

};

}

#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK
#endif // LANCZOS_EIGEN_SOLVER_H_
