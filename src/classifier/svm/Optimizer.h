#ifndef __OPTIMIZER__H__
#define __OPTIMIZER__H__

#include "lib/common.h"

typedef struct quadratic_program {
  LONG   opt_n;            /* number of variables */
  LONG   opt_m;            /* number of linear equality constraints */
  double *opt_ce,*opt_ce0; /* linear equality constraints */
  double *opt_g;           /* hessian of objective */
  double *opt_g0;          /* linear part of objective */
  double *opt_xinit;       /* initial value for variables */
  double *opt_low,*opt_up; /* box constraints */
} QP;
    
/* interface to QP-solver */
double *optimize_qp(QP *qp,double *epsilon_crit, long nx,double *threshold, long& svm_maxqpsize);
#endif
