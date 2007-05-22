/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2007 Soeren Sonnenburg
 * Copyright (C) 2006-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef CCPLEX_H__
#define CCPLEX_H__

#include "lib/config.h"

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}

#include "lib/common.h"
#include "base/SGObject.h"

enum E_PROB_TYPE
{
	LINEAR,
	QP
};

class CCplex : public CSGObject
{
public:

	CCplex();
	~CCplex();

	bool init(E_PROB_TYPE t);
	bool cleanup();

	/// call this to setup linear part
	///
	/// setup lp, to minimize
	/// objective[0]*x_0 ... objective[cols-1]*x_{cols-1}
	/// w.r.t. x
	/// s.t. constraint_mat*x <= rhs
	/// lb[i] <= x[i] <= ub[i] for all i
	bool setup_lp(DREAL* objective, DREAL* constraints_mat, INT rows, INT cols, DREAL* rhs, DREAL* lb, DREAL* ub);

	/// call this to setup quadratic part H
	/// x'*H*x
	/// call setup_lp before (to setup the linear part / linear constraints)
	bool setup_qp(DREAL* H, INT dim);
	bool optimize(DREAL* sol, INT dim);

	bool dense_to_cplex_sparse(DREAL* H, INT rows, INT cols, int* &qmatbeg, int* &qmatcnt, int* &qmatind, double* &qmatval);
protected:
  CPXENVptr     env;
  CPXLPptr      lp;
  bool          lp_initialized;

  E_PROB_TYPE problem_type;
};
#endif
#endif
