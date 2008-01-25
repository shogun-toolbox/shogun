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

#include <math.h>
#include <limits.h>

#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/config.h"
#include "lib/common.h"
#include "kernel/Kernel.h"

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

/** clas QPBSVMLib */
class CQPBSVMLib: public CSGObject
{
	public:
		/** constructor
		 *
		 * @param H symmetric matrix of size n x n
		 * @param n size of H's matrix
		 * @param f is vector of size m
		 * @param m size of vector f
		 * @param UB UB
		 */
		CQPBSVMLib(DREAL* H, INT n, DREAL* f, INT m, DREAL UB=1.0);

		/// result has to be allocated & zeroed
		INT solve_qp(DREAL* result, INT len);

		/** set solver
		 *
		 * @param solver new solver
		 */
		inline void set_solver(E_QPB_SOLVER solver)
		{
			m_solver=solver;
		}

		~CQPBSVMLib();

	protected:
		/** get col
		 *
		 * @param col col to get
		 * @return col indexed by col
		 */
		inline DREAL* get_col(INT col)
		{
			return &m_H[m_dim*col];
		}

		/** Usage: exitflag = qpbsvm_sca(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_sca(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		/** Usage: exitflag = qpbsvm_scas(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_scas(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		/** Usage: exitflag = qpbsvm_scamv(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_scamv(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		/** Usage: exitflag = qpbsvm_prloqo(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_prloqo(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		/** Usage: exitflag = qpbsvm_gauss_seidel(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_gauss_seidel(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		/** Usage: exitflag = qpbsvm_gradient_descent(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_gradient_descent(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
#ifdef USE_CPLEX
		/** Usage: exitflag = qpbsvm_cplex(UB, dim, tmax,
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_cplex(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
#endif

	protected:
		/** matrix H */
		DREAL* m_H;
		/** diagonal of H */
		DREAL* m_diag_H;
		/** dim */
		INT m_dim;

		/** vector f */
		DREAL* m_f;

		/** UB */
		DREAL m_UB;

		/** tmax */
		INT m_tmax;
		/** tolabs */
		DREAL m_tolabs;
		/** tolrel */
		DREAL m_tolrel;
		/** tolKKT */
		DREAL m_tolKKT;
		/** solver */
		E_QPB_SOLVER m_solver;
};
#endif //QPBSVMLIB_H__
