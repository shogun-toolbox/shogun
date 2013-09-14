/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef ITERATIVE_SHIFTED_LINEAR_FAMILY_SOLVER_H_
#define ITERATIVE_SHIFTED_LINEAR_FAMILY_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>

namespace shogun
{
template <class T> class SGVector;
template <class T> class CLinearOperator;

/** 
 * @brief abstract template base for CG based solvers to the solution of 
 * shifted linear systems of the form \f$(A+\sigma)x=b\f$ for several values
 * of \f$\sigma\f$ simultaneously, using only as many matrix-vector operations
 * as the solution of a single system requires. This class adds another
 * interface to the basic iterative linear solver that takes the shifts,
 * \f$\sigma\f$, and also weights, \f$\alpha\f$, and returns the summation
 * \f$\sum_{i} \alpha_{i}x_{i}\f$, where \f$x_{i}\f$ is the solution of the
 * system \f$(A+\sigma_{i})x_{i}=b\f$.
 *
 * Reference: Beat Jegerlehner, Krylov space solvers for shifted linear
 * systems, 1996.
 */
template<class T, class ST=T> class CIterativeShiftedLinearFamilySolver : public CIterativeLinearSolver<T, T>
{

public:
	/** default constructor */
	CIterativeShiftedLinearFamilySolver();

	/** destructor */
	virtual ~CIterativeShiftedLinearFamilySolver();

	/** 
	 * abstract solve method for solving real linear systems which computes
	 * the solution for non-shifted linear system in CG iteration
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @return the solution vector
	 */
	virtual SGVector<T> solve(CLinearOperator<T>* A, SGVector<T> b) = 0;

	/**
	 * abstract method that solves the shifted family of linear systems, multiples
	 * each solution vector with a weight, computes a summation over all the
	 * shifts and returns the final solution vector
	 *
	 * @param A the linear operator of the system
	 * @param b the vector of the system
	 * @param shifts the shifts of the shifted system
	 * @param weights the weights to be multiplied with each solution for each
	 * shift
	 */
	virtual SGVector<ST> solve_shifted_weighted(CLinearOperator<T>* A,
		SGVector<T> b, SGVector<ST> shifts, SGVector<ST> weights) = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "IterativeShiftedLinearFamilySolver";
	}

protected:
	/**
	 * compute \f$\zeta^{\sigma}_{n+1}\f$ as \f$\frac{\zeta^{\sigma}_{n}
	 * \zeta^{\sigma}_{n-1}\beta_{n-1}}{\beta_{n}\alpha{n}(\zeta^{\sigma}_{n-1}
	 * -\zeta^{\sigma}_{n}+\zeta^{\sigma}_{n-1}\beta_{n-1}(1-\sigma\beta_{n}}\f$
	 * [see Jergerlehner, eq 2.44]
	 *
	 * @param zeta_sh_old \f$\zeta^{\sigma}_{n-1}\f$ shifted params
	 * @param zeta_sh_cur \f$\zeta^{\sigma}_{n}\f$ shifted params
	 * @param shifts \f$\sigma\f$ shifts
	 * @param beta_old \f$\beta_{n-1}\f$, non-shifted
	 * @param beta_cur \f$\beta_{n}\f$, non-shifted
	 * @param alpha \f$\alpha\f$ non-shifted
	 * @param zeta_sh_new \f$\zeta^{\sigma}_{n+1}\f$ to be computed
	 */
	void compute_zeta_sh_new(const SGVector<ST>& zeta_sh_old,
		const SGVector<ST>& zeta_sh_cur, const SGVector<ST>& shifts,
		const T& beta_old, const T& beta_cur, const T& alpha, SGVector<ST>& zeta_sh_new);

	/**
	 * compute \f$\beta^{\sigma}_{n}\f$ as \f$\beta_{n}\frac{\zeta^{\sigma}_{n+1}}
	 * {\zeta^{\sigma}_{n}}\f$
	 *
	 * @param zeta_sh_new \f$\zeta^{\sigma}_{n+1}\f$ shifted params
	 * @param zeta_sh_cur \f$\zeta^{\sigma}_{n}\f$ shifted params
	 * @param beta_cur \f$\beta_{n}\f$, non-shifted
	 * @param beta_sh \f$\beta^{\sigma}_{n}\f$, to be computed
	 */
	void compute_beta_sh(const SGVector<ST>& zeta_sh_new,
		const SGVector<ST>& zeta_sh_cur, const T& beta_cur, SGVector<ST>& beta_sh);

	/**
	 * compute \f$alpha^{\sigma}_{n}\f$ as \f$\alpha_{n}\frac{\zeta^{\sigma}
	 * _{n}\beta^{\sigma}_{n-1}}{\zeta^{\sigma}_{n-1}\beta_{n-1}}\f$
	 *
	 * @param zeta_sh_cur \f$\zeta^{\sigma}_{n}\f$ shifted params
	 * @param zeta_sh_old \f$\zeta^{\sigma}_{n-1}\f$ shifted params
	 * @param beta_sh_old \f$\beta^{\sigma}_{n-1}\f$, shifted params
	 * @param beta_old \f$\beta_{n-1}\f$, non-shifted
	 * @param alpha \f$\alpha_{n}\f$, non-shifted
	 * @param alpha_sh \f$\alpha^{\sigma}_{n}\f$, to be computed
	 */
	void compute_alpha_sh(const SGVector<ST>& zeta_sh_cur,
		const SGVector<ST>& zeta_sh_old, const SGVector<ST>& beta_sh_old,
		const T& beta_old, const T& alpha, SGVector<ST>& alpha_sh);

};

}

#endif // ITERATIVE_SHIFTED_LINEAR_FAMILY_SOLVER_H_
