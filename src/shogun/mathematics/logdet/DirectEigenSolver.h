/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */
#ifndef DIRECT_EIGEN_SOLVER_H_
#define DIRECT_EIGEN_SOLVER_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/logdet/EigenSolver.h>

namespace shogun
{
template<class T> class CLinearOperator;

/** @brief Class that computes eigenvalues of a real valued, self-adjoint
 * dense matrix linear operator using Eigen3
 */
class CDirectEigenSolver : public CEigenSolver
{
public:
	/** default constructor */
	CDirectEigenSolver();

	/** destructor */
	virtual ~CDirectEigenSolver();

	/** 
	 * compute method for computing eigenvalues of a real valued dense matrix
	 * linear operator
	 *
	 * @param A the linear operator whose eigenvalues are to be computed
	 */
	virtual void compute(CLinearOperator<float64_t>* A);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CDirectEigenSolver";
	}
};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_EIGEN_SOLVER_H_
