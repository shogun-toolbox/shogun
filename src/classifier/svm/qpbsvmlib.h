/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library for solving QP task required for learning SVM without bias term. 
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
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

class CQPBSVMLib: public CSGObject
{
	public:
		/// H is a symmetric matrix of size n x n
		/// f is vector of size m
		CQPBSVMLib(DREAL* H, INT n, DREAL* f, INT m, DREAL UB=1.0);

		/// result has to be allocated & zeroed
		INT solve_qp(DREAL* result, INT len);

		inline void set_solver(E_QPB_SOLVER solver)
		{
			m_solver=solver;
		}

		~CQPBSVMLib();

	protected:
		inline DREAL* get_col(INT col)
		{
			return &m_H[m_dim*col];
		}

		/** Usage: exitflag = qpbsvm_sca(UB, dim, tmax, 
		tolabs, tolrel, tolKKT, x, Nabla, &t, &History, verb ) */
		INT qpbsvm_sca(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		INT qpbsvm_scas(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		INT qpbsvm_scamv(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		INT qpbsvm_prloqo(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		INT qpbsvm_gauss_seidel(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
		INT qpbsvm_gradient_descent(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
#ifdef USE_CPLEX
		INT qpbsvm_cplex(DREAL *x, DREAL *Nabla, INT *ptr_t, DREAL **ptr_History, INT verb);
#endif

	protected:
		DREAL* m_H;
		DREAL* m_diag_H;
		INT m_dim;

		DREAL* m_f;

		DREAL m_UB;

		INT m_tmax;
		DREAL m_tolabs;
		DREAL m_tolrel;
		DREAL m_tolKKT;
		E_QPB_SOLVER m_solver;
};
#endif //QPBSVMLIB_H__
