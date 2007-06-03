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

#include "features/SparseFeatures.h"
#include "features/Labels.h"

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

	/// init cplex with problem type t and retry timeout 60 seconds
	bool init(E_PROB_TYPE t, INT timeout=60);
	bool cleanup();

	bool setup_lpboost(DREAL C, INT num_cols);
	bool add_lpboost_constraint(TSparseEntry<DREAL>* h, INT len, INT ulen, CLabels* label);

	/// given N sparse inputs x_i, and corresponding labels y_i i=0...N-1
	/// create the following 1-norm SVM problem & transfer to cplex
	/// 
	////////////////////////////////////////////////////////////////// 
	/// min_w 		sum_{i=0}^N ( w^+_i + w^-_i) C \sum_{i=0}^N \xi_i
	/// w=[w^+ w^-]
	/// b, xi
	/// 
	/// -y_i(+w_+^T x_i + b)-xi_i <= -1
	/// -y_i(-w_-^T x_i + b)-xi_i <= -1
	/// x_i >= 0 
	/// w_i >= 0    forall i=1...N
	////////////////////////////////////////////////////////////////// 
	/// min f^x
	/// Ax <= b
	/// -x <= 0
	/// 
	/// lb= [ -inf, //b
	/// 	  2*dims [0], //w
	/// 	  num_train [0] //xi 
	/// 	]
	/// 
	/// ub= [ inf, //b
	/// 	  2*dims [inf], //w
	/// 	  num_train [inf] //xi 
	/// 	]
	/// 
	/// f= [0,2*dim[1], num_train*C]
	/// A= [-y', // b
	/// 	-y_ix_i // w_+
	/// 	+y_ix_i // w_-
	/// 	-1 //xi
	/// 	]
	/// 
	/// 	dim(A)=(n,1+2*dim+n)
	/// 
	/// b =  -1 -1 -1 -1 ... 
	bool setup_lpm(DREAL C, CSparseFeatures<DREAL>* x, CLabels* y, bool use_bias);

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
	bool optimize(DREAL* sol, DREAL* lambda=NULL);

	bool dense_to_cplex_sparse(DREAL* H, INT rows, INT cols, int* &qmatbeg, int* &qmatcnt, int* &qmatind, double* &qmatval);
protected:
  CPXENVptr     env;
  CPXLPptr      lp;
  bool          lp_initialized;

  E_PROB_TYPE problem_type;
};
#endif
#endif
