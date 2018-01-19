/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soumyajit De, Björn Esser
 */

#ifndef DIRECT_EIGEN_SOLVER_H_
#define DIRECT_EIGEN_SOLVER_H_

#include <shogun/lib/config.h>

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

#endif // DIRECT_EIGEN_SOLVER_H_
