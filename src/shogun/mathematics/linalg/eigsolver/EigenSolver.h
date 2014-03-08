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
#include <shogun/mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{

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
	}

	/** destructor */
	virtual ~CEigenSolver()
	{
	}

	/**
	 * abstract compute method for computing eigenvalues of a real
	 * valued linear operator
	 *
	 * @param linear_operator real valued self-adjoint linear operator
	 * whose eigenvalues have to be found
	 */
	virtual void compute(CLinearOperator<float64_t>* linear_operator) = 0;

	/** sets the min eigelvalue of a real valued self-adjoint linear operator */
	void set_min_eigenvalue(float64_t min_eigenvalue)
	{
		m_min_eigenvalue=min_eigenvalue;
		m_is_computed_min=true;
	}

	/** @return min eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_min_eigenvalue() const
	{
		return m_min_eigenvalue;
	}

	/** sets the max eigelvalue of a real valued self-adjoint linear operator */
	void set_max_eigenvalue(float64_t max_eigenvalue)
	{
		m_max_eigenvalue=max_eigenvalue;
		m_is_computed_max=true;
	}

	/** @return max eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_max_eigenvalue() const
	{
		return m_max_eigenvalue;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "EigenSolver";
	}
protected:
	/** min eigenvalue */
	float64_t m_min_eigenvalue;

	/** max eigenvalue */
	float64_t m_max_eigenvalue;

	/** flag that denotes that the minimum eigenvalue is already computed */
	bool m_is_computed_min;

	/** flag that denotes that the maximum eigenvalue is already computed */
	bool m_is_computed_max;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_min_eigenvalue=0.0;
		m_max_eigenvalue=0.0;
		m_is_computed_min=false;
		m_is_computed_max=false;

		SG_ADD(&m_min_eigenvalue, "min_eigenvalue",
			"Minimum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_max_eigenvalue, "max_eigenvalue",
			"Maximum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_is_computed_min, "is_computed_min",
			"Flag denoting that the minimum eigenvalue has already been computed",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_max_eigenvalue, "is_computed_max",
			"Flag denoting that the maximum eigenvalue has already been computed",
			MS_NOT_AVAILABLE);

	}
};

}

#endif // EIGEN_SOLVER_H_
