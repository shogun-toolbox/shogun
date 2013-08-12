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
template<class T> class CDenseMatrixOperator;

/** @brief Class that computes eigenvalues of a real valued, self-adjoint
 * dense matrix linear operator using Eigen3
 */
class CDirectEigenSolver : public CEigenSolver
{
public:
	/** default constructor */
	CDirectEigenSolver();

	/** 
	 * constructor
	 * 
	 * @param linear_operator self-adjoint dense-matrix linear operator whose
	 * eigenvalues have to be found
	 */
	CDirectEigenSolver(CDenseMatrixOperator<float64_t>* linear_operator);

	/** destructor */
	virtual ~CDirectEigenSolver();

	/** 
	 * compute method for computing eigenvalues of a real valued dense matrix
	 * linear operator
	 */
	virtual void compute();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectEigenSolver";
	}
};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_EIGEN_SOLVER_H_
