/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */
#ifndef EIGEN_SOLVER_H_
#define EIGEN_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
template<class T> class CLinearOperator;

/** @brief Abstract base class that provides an abstract compute method for
 * computing eigenvalues of a real valued, self-adjoint linear operator. It 
 * also provides method for getting min and max eigenvalues.
 */
class CEigenSolver : public CSGObject
{
public:
	/** default constructor */
	CEigenSolver()
	: CSGObject()
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CEigenSolver()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/** 
	 * abstract compute method for computing eigenvalues of a real
	 * valued linear operator
	 *
	 * @param A the linear operator whose eigenvalues are to be computed
	 */
	virtual void compute(CLinearOperator<float64_t>* A) = 0;

	/** @return min eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_min_eigenvalue() const
	{
		return m_min_eigenvalue;
	}

	/** @return max eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_max_eigenvalue() const
	{
		return m_max_eigenvalue;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CEigenSolver";
	}
protected:
	/** min eigenvalue */
	float64_t	m_min_eigenvalue;

	/** max eigenvalue */
	float64_t	m_max_eigenvalue;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_min_eigenvalue=0.0;
		m_max_eigenvalue=0.0;
	
		SG_ADD(&m_min_eigenvalue, "min_eigenvalue",
			"Minimum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_max_eigenvalue, "max_eigenvalue",
			"Maximum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);
	}
};

}

#endif // EIGEN_SOLVER_H_
