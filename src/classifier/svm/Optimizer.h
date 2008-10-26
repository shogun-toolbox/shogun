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
  double *opt_ce;
  /** linear equality constraint */
  double *opt_ce0;
  /** hessian of objective */
  double *opt_g;
  /** linear part of objective */
  double *opt_g0;
  /** initial value for variables */
  double *opt_xinit;
  /** low box constraint */
  double *opt_low;
  /** up box constraint */
  double *opt_up;
} QP;

/* interface to QP-solver */
double *optimize_qp(QP *qp,double *epsilon_crit, int32_t nx,double *threshold, int32_t& svm_maxqpsize);
#endif
