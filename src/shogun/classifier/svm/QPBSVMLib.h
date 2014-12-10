/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Library for solving QP task required for learning SVM without bias term.
 *
 * Written (W) 1999-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2008 Center for Machine Perception, CTU FEL Prague
 *
 -------------------------------------------------------------------- */

#ifndef QPBSVMLIB_H__
#define QPBSVMLIB_H__

#include <limits.h>

#include <shogun/mathematics/Math.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum E_QPB_SOLVER
{
	QPB_SOLVER_SCA,	// sequential coordinate wise (gaussian seidel based)
	QPB_SOLVER_SCAS,	// sequential coordinate wise selecting the variable
	// gaining 'best' improved
	QPB_SOLVER_SCAMV, // sequential coordinate wise selecting variable most violating kkt's
	QPB_SOLVER_PRLOQO,// via pr_loqo
	QPB_SOLVER_CPLEX,  // via cplex
	QPB_SOLVER_GS,  // gaussian seidel
	QPB_SOLVER_GRADDESC  // gaussian seidel
};
#endif

/** @brief class QPBSVMLib */
class CQPBSVMLib: public CSGObject
{
	public:
		/** default constructor  */
		CQPBSVMLib();

		/** constructor
		 *
		 * @param H symmetric matrix of size n x n
		 * @param n size of H's matrix
		 * @param f is vector of size m
		 * @param m size of vector f
		 * @param UB UB
		 */
		CQPBSVMLib(
			float64_t* H, int32_t n, float64_t* f, int32_t m, float64_t UB=1.0);

		/// result has to be allocated & zeroed
		int32_t solve_qp(float64_t* result, int32_t len);

		/** set solver
		 *
		 * @param solver new solver
		 */
		inline void set_solver(E_QPB_SOLVER solver)
		{
			m_solver=solver;
		}

		virtual ~CQPBSVMLib();

	protected:
		/** get col
		 *
		 * @param col col to get
		 * @return col indexed by col
		 */
		inline float64_t* get_col(int32_t col)
		{
			return &m_H[m_dim*col];
		}

		/** Usage: exitflag = qpbsvm_sca(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_sca(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
		/** Usage: exitflag = qpbsvm_scas(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_scas(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
		/** Usage: exitflag = qpbsvm_scamv(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_scamv(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
		/** Usage: exitflag = qpbsvm_prloqo(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_prloqo(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
		/** Usage: exitflag = qpbsvm_gauss_seidel(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_gauss_seidel(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
		/** Usage: exitflag = qpbsvm_gradient_descent(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_gradient_descent(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
#ifdef USE_CPLEX
		/** Usage: exitflag = qpbsvm_cplex(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		int32_t qpbsvm_cplex(
			float64_t *x, float64_t *Nabla, int32_t *ptr_t,
			float64_t **ptr_History, int32_t verb);
#endif

		/** @return object name */
		inline const char* get_name() const { return "QPBSVMLib"; }

	protected:
		/** matrix H */
		float64_t* m_H;
		/** diagonal of H */
		float64_t* m_diag_H;
		/** dim */
		int32_t m_dim;

		/** vector f */
		float64_t* m_f;

		/** UB */
		float64_t m_UB;

		/** tmax */
		int32_t m_tmax;
		/** tolabs */
		float64_t m_tolabs;
		/** tolrel */
		float64_t m_tolrel;
		/** tolKKT */
		float64_t m_tolKKT;
		/** solver */
		E_QPB_SOLVER m_solver;
};
}
#endif //QPBSVMLIB_H__
