/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#ifndef CG_M_SHIFTED_FAMILY_SOLVER_H_
#define CG_M_SHIFTED_FAMILY_SOLVER_H_

#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>

namespace shogun
{
template<class T> class LinearOperator;
template<class T> class SGVector;

/**
 * @brief class that uses conjugate gradient method for solving a shifted
 * linear system family where the linear opeator is real valued and symmetric
 * positive definite, the vector is real valued, but the shifts are complex
 *
 * Note: The implementation of solve_shifted_weighted has been adapted from the
 * open source library Krylstat (https://github.com/Froskekongen/KRYLSTAT/),
 * written by Erlend Aune, under GPL2+
 */
class CGMShiftedFamilySolver
 : public IterativeShiftedLinearFamilySolver<float64_t, complex128_t>
{

public:
	/** default constructor */
	CGMShiftedFamilySolver();

	/** one arg constructor */
	CGMShiftedFamilySolver(bool store_residuals);

	/** destructor */
	virtual ~CGMShiftedFamilySolver();

	/**
	 * solve method for solving linear systems assuming no shift
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<float64_t> solve(std::shared_ptr<LinearOperator<float64_t>> A,
		SGVector<float64_t> b);

	/**
	 * method that solves the shifted family of linear systems, multiples
	 * each solution vector with a weight, computes a summation over all the
	 * shifts and returns the final solution vector
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @param shifts the shifts of the shifted system
	 * @param weights the weights to be multiplied with each solution for each
	 * shift
	 */
	virtual SGVector<complex128_t> solve_shifted_weighted(
		std::shared_ptr<LinearOperator<float64_t>> A, SGVector<float64_t> b,
		SGVector<complex128_t> shifts, SGVector<complex128_t> weights,
		bool negate = false);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CGMShiftedFamilySolver";
	}

};

}

#endif // CG_M_SHIFTED_FAMILY_SOLVER_H_
