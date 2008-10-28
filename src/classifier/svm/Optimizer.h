/***********************************************************************/
/*                                                                     */
/*   Optimizer.h                                                       */
/*                                                                     */
/*   Interface to the PR_LOQO optimization package for SVM.            */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 19.07.99                                                    */
/*                                                                     */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved   */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/
#ifndef __OPTIMIZER__H__
#define __OPTIMIZER__H__

#include "lib/common.h"
# define DEF_PRECISION 1E-14

/** quadratic program */
typedef struct quadratic_program {
  /** number of variables */
  int32_t   opt_n;
  /** number of linear equality constraints */
  int32_t   opt_m;
  /** linear equality constraint */
  float64_t *opt_ce;
  /** linear equality constraint */
  float64_t *opt_ce0;
  /** hessian of objective */
  float64_t *opt_g;
  /** linear part of objective */
  float64_t *opt_g0;
  /** initial value for variables */
  float64_t *opt_xinit;
  /** low box constraint */
  float64_t *opt_low;
  /** up box constraint */
  float64_t *opt_up;
} QP;

/* interface to QP-solver */
float64_t *optimize_qp(
	QP *qp,float64_t *epsilon_crit, int32_t nx,float64_t *threshold,
	int32_t& svm_maxqpsize);
#endif
