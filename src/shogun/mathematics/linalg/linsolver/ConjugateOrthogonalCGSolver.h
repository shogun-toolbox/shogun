/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#ifndef CONJUGATE_ORTHOGONAL_CG_SOLVER_H_
#define CONJUGATE_ORTHOGONAL_CG_SOLVER_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>

namespace shogun
{
template<class T> class LinearOperator;
template<class T> class SGVector;

/**
 * @brief class that uses conjugate orthogonal conjugate gradient method of
 * solving a linear system involving a complex valued linear operator and
 * vector. Useful for large sparse systems involving sparse symmetric matrices
 * that are not Herimitian.
 *
 * Reference: Vorst, Melissen, "A Petrov-Galerkin Type Method for Solving Ax=b,
 * Where A Is Symmetric Complex". IEEE Transactions on Magnetics, Vol. 26,
 * No. 2, March 1990
 */
class ConjugateOrthogonalCGSolver
 : public IterativeLinearSolver<complex128_t, float64_t>
{

public:
	/** default constructor */
	ConjugateOrthogonalCGSolver();

	/** one arg constructor */
	ConjugateOrthogonalCGSolver(bool store_residuals);

	/** destructor */
	virtual ~ConjugateOrthogonalCGSolver();

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
		return "ConjugateOrthogonalCGSolver";
	}

};

}

#endif // CONJUGATE_ORTHOGONAL_CG_SOLVER_H_
