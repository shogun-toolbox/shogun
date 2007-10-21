/*-----------------------------------------------------------------------
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Library of solvers for QP task required in StructSVM learning.
 *
 * Written (W) 1999-2007 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 1999-2007 Center for Machine Perception, CTU FEL Prague 
 *
-------------------------------------------------------------- */

#ifndef QPSSVMLIB_H__ 
#define QPSSVMLIB_H__ 

#include "lib/config.h"
#include "lib/common.h"
#include "base/SGObject.h"

enum E_QPS_SOLVER
{
	QPS_SOLVER_IMDM,
	QPS_SOLVER_SOLVER,
};

class CQPSSVMLib: public CSGObject
{
	public:
		/// H is a symmetric matrix of size n x n
		/// f is vector of size m
		CQPSSVMLib(DREAL* H, WORD* I, DREAL* f, INT n, INT m);
		~CQPSSVMLib();

		inline INT solve_qp(DREAL* result, INT len)
		{
			switch (m_solver)
			{
				case QPS_SOLVER_IMDM:
					return qpssvm_imdm(result, NULL, NULL);
					break;
				case QPS_SOLVER_SOLVER:
					return qpssvm_solver(result);
					break;
			}
		}

		inline DREAL get_primal()
		{
			return m_QP;
		}

		inline DREAL get_dual()
		{
			return m_QD;
		}

		inline void set_solver(E_QPS_SOLVER solver)
		{
			m_solver=solver;
		}
		
	protected:
		INT qpssvm_imdm(DREAL *x, INT *ptr_t, DREAL **ptr_History);
		INT qpssvm_solver(DREAL *x);

		inline DREAL* get_col(INT col)
		{
			return &m_H[m_dim*col];
		}

	protected:
		DREAL* m_H;
		WORD* m_I;
		DREAL* m_diag_H;
		DREAL* m_f;

		INT m_var;
		INT m_dim;
		DREAL m_b;

		E_QPS_SOLVER m_solver;

		DREAL m_QP;
		DREAL m_QD;

		INT m_tmax;
		DREAL m_tolabs;
		DREAL m_tolrel;

		INT m_verb;
};

#endif //QPBSVMLIB_H__
