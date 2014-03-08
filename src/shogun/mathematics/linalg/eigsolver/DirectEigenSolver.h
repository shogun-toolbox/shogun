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
#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>

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

	/** destructor */
	virtual ~CDirectEigenSolver();

	/**
	 * compute method for computing eigenvalues of a real valued dense matrix
	 * linear operator
	 *
	 * @param linear_operator real valued self-adjoint linear operator
	 * whose eigenvalues have to be found
	 */
	virtual void compute(CLinearOperator<float64_t>* linear_operator);

	/**
	 * compute method for computing eigenvalues of a real valued dense matrix
	 *
	 * @param m real valued self-adjoint matrix whose eigenvalues have to be
	 * found
	 */
	void compute(SGMatrix<float64_t> m);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectEigenSolver";
	}
};

}

#endif // HAVE_EIGEN3
#endif // DIRECT_EIGEN_SOLVER_H_
