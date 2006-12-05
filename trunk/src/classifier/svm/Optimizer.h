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

typedef struct quadratic_program {
  INT   opt_n;            /* number of variables */
  INT   opt_m;            /* number of linear equality constraints */
  double *opt_ce,*opt_ce0; /* linear equality constraints */
  double *opt_g;           /* hessian of objective */
  double *opt_g0;          /* linear part of objective */
  double *opt_xinit;       /* initial value for variables */
  double *opt_low,*opt_up; /* box constraints */
} QP;
    
/* interface to QP-solver */
double *optimize_qp(QP *qp,double *epsilon_crit, INT nx,double *threshold, INT& svm_maxqpsize);
#endif
