/************************************************************************/
/*                                                                      */
/*   svm_common.h                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 01.11.00                                                     */
/*                                                                      */
/*   Copyright (c) 2000  Universitaet Dortmund - All rights reserved    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#ifndef _SVM_COMMON_H__
#define _SVM_COMMON_H__

# include <stdio.h>
# include <ctype.h>
# include <math.h>
# include <string.h>
# include <stdlib.h>
# include <time.h> 
# include <float.h>

# define VERSION       "V3.50"
# define VERSION_DATE  "01.11.00"

# define CFLOAT  float       /* the type of float to use for caching */
                             /* kernel evaluations. Using float saves */
                             /* us some memory, but you can use double, too */
# define FNUM    long        /* the type used for storing feature ids */
# define FVAL    float       /* the type used for storing feature values */

# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynoial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */
# define TOP	 4           /* top kernel type */
# define LINEAR_TOP  5       /* linear top kernel type */

/*
typedef struct word {
  FNUM    wnum;	
  FVAL    weight;
} WORD;*/


typedef struct doc {
  long    docnum;
  double  twonorm_sq;
//  WORD    *words;
} DOC;




typedef struct learn_parm {
  double svm_c;                /* upper bound C on alphas */
  double svm_costratio;        /* factor to multiply C for positive examples */
  double transduction_posratio;/* fraction of unlabeled examples to be */
                               /* classified as positives */
  long   biased_hyperplane;    /* if nonzero, use hyperplane w*x+b=0 
				  otherwise w*x=0 */
  long   svm_maxqpsize;        /* size q of working set */
  long   svm_newvarsinqp;      /* new variables to enter the working set 
				  in each iteration */
  double epsilon_crit;         /* tolerable error for distances used 
				  in stopping criterion */
  double epsilon_shrink;       /* how much a multiplier should be above 
				  zero for shrinking */
  long   svm_iter_to_shrink;   /* iterations h after which an example can
				  be removed by shrinking */
  long   remove_inconsistent;  /* exclude examples with alpha at C and 
				  retrain */
  long   skip_final_opt_check; /* do not check KT-Conditions at the end of
				  optimization for examples removed by 
				  shrinking. WARNING: This might lead to 
				  sub-optimal solutions! */
  long   compute_loo;          /* if nonzero, computes leave-one-out
				  estimates */
  double rho;                  /* parameter in xi/alpha-estimates and for
				  pruning leave-one-out range [1..2] */
  long   xa_depth;             /* parameter in xi/alpha-estimates upper
				  bounding the number of SV the current
				  alpha_t is distributed over */
  char predfile[200];          /* file for predicitions on unlabeled examples
				  in transduction */
  char alphafile[200];         /* file to store optimal alphas in. use  
				  empty string if alphas should not be 
				  output */

  /* you probably do not want to touch the following */
  double epsilon_const;        /* tolerable error on eq-constraint */
  double epsilon_a;            /* tolerable error on alphas at bounds */
  double opt_precision;        /* precision of solver, set to e.g. 1e-21 
				  if you get convergence problems */

  /* the following are only for internal use */
  long   svm_c_steps;          /* do so many steps for finding optimal C */
  double svm_c_factor;         /* increase C by this factor every step */
  double svm_costratio_unlab;
  double svm_unlabbound;
  double *svm_cost;            /* individual upper bounds for each var */
////  long   totwords;             /* number of features */
} LEARN_PARM;

typedef struct kernel_parm {
  long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid, 4=custom */
  long    poly_degree;
  double  rbf_gamma;
  double  coef_lin;
  double  coef_const;
  char    custom[50];    /* for user supplied kernel */
} KERNEL_PARM;

typedef struct model {
  long    sv_num;	
  long    at_upper_bound;
  double  b;
  DOC     **supvec;
  double  *alpha;
  long    *index;       /* index from docnum to position in model */
////  long    totwords;     /* number of features */
  long    totdoc;       /* number of training documents */
  KERNEL_PARM kernel_parm; /* kernel */

  /* the following values are not written to file */
  double  loo_error,loo_recall,loo_precision; /* leave-one-out estimates */
  double  xa_error,xa_recall,xa_precision;    /* xi/alpha estimates */
////  double  *lin_weights;                       /* weights for linear case using folding */
} MODEL;

typedef struct quadratic_program {
  long   opt_n;            /* number of variables */
  long   opt_m;            /* number of linear equality constraints */
  double *opt_ce,*opt_ce0; /* linear equality constraints */
  double *opt_g;           /* hessian of objective */
  double *opt_g0;          /* linear part of objective */
  double *opt_xinit;       /* initial value for variables */
  double *opt_low,*opt_up; /* box constraints */
} QP;

typedef struct kernel_cache {
  long   *index;  /* cache some kernel evalutations */
  CFLOAT *buffer; /* to improve speed */
  long   *invindex;
  long   *active2totdoc;
  long   *totdoc2active;
  long   *lru;
  long   *occu;
  long   elems;
  long   max_elems;
  long   time;
  long   activenum;
  long   buffsize;
} KERNEL_CACHE;


typedef struct timing_profile {
  long   time_kernel;
  long   time_opti;
  long   time_shrink;
  long   time_update;
  long   time_model;
  long   time_check;
  long   time_select;
} TIMING;

typedef struct shrink_state {
  long   *active;
  long   *inactive_since;
  long   deactnum;
  double **a_history;
} SHRINK_STATE;

double classify_example(MODEL *, DOC *);
double classify_example_linear(MODEL *, DOC *);
CFLOAT kernel(KERNEL_PARM *, DOC *, DOC *); 
double custom_kernel(KERNEL_PARM *, DOC *, DOC *); 
////double sprod_ss(WORD *, WORD *);
double model_length_s(MODEL *, KERNEL_PARM *);
void   clear_vector_n(double *, long);
////void   add_vector_ns(double *, WORD *, double);
////double sprod_ns(double *, WORD *);
void   add_weight_vector_to_linear_model(MODEL *);
void   read_model(char *, MODEL *, long, long);
void   read_documents(char *, DOC *, long *, long, long, long *, long *);
int    parse_document(char *, DOC *, long *, long *, long);
void   nol_ll(char *, long *, long *, long *);
long   minl(long, long);
long   maxl(long, long);
long   get_runtime();
void   *my_malloc(long); 
void   copyright_notice();
# ifdef _WIN32
   int isnan(double);
# endif

extern long   verbosity;              /* verbosity level (0-4) */
extern long   kernel_cache_statistic;
#endif