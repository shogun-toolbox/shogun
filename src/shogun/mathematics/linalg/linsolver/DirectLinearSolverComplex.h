/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#ifndef DIRECT_LINEAR_SOLVER_COMPLEX_H_
#define DIRECT_LINEAR_SOLVER_COMPLEX_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>

namespace shogun
{

/** solver type for direct solvers */
enum EDirectSolverType
{
	DS_LLT=0,
	DS_QR_NOPERM=1,
	DS_QR_COLPERM=2,
	DS_QR_FULLPERM=3,
	DS_SVD=4
};

/** @brief Class that provides a solve method for complex dense-matrix
 * linear systems
 */
class DirectLinearSolverComplex : public LinearSolver<complex128_t, float64_t>
{
public:
	/** default constructor */
	DirectLinearSolverComplex();

	/**
	 * constructor
	 *
	 * @param type the type of solver to be used in solve method
	 */
	DirectLinearSolverComplex(EDirectSolverType type);

	/** destructor */
	virtual ~DirectLinearSolverComplex();

	/**
	 * solve method for solving complex linear systems
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<complex128_t> solve(std::shared_ptr<LinearOperator<complex128_t>> A,
			SGVector<float64_t> b);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DirectLinearSolverComplex";
	}

private:
	/** the type of solver to be used in solve method */
	const EDirectSolverType m_type;

};

}

#endif // DIRECT_LINEAR_SOLVER_COMPLEX_H_
