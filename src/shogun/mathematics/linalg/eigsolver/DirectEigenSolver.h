/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soumyajit De, Bjoern Esser
 */

#ifndef DIRECT_EIGEN_SOLVER_H_
#define DIRECT_EIGEN_SOLVER_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>

namespace shogun
{
template<class T> class DenseMatrixOperator;

/** @brief Class that computes eigenvalues of a real valued, self-adjoint
 * dense matrix linear operator using Eigen3
 */
class DirectEigenSolver : public EigenSolver
{
public:
	/** default constructor */
	DirectEigenSolver();

	/**
	 * constructor
	 *
	 * @param linear_operator self-adjoint dense-matrix linear operator whose
	 * eigenvalues have to be found
	 */
	DirectEigenSolver(const std::shared_ptr<DenseMatrixOperator<float64_t>>& linear_operator);

	/** destructor */
	virtual ~DirectEigenSolver();

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
