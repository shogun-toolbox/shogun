#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef SUNOS
#define bool int
#define false 0
#define true 1
#endif

#include "intpoint/intpoint.h"

/**@name Standard Types 
 * Definition of Platform independent Types
*/
//@{
/// Type WORD is 2 bytes in size
typedef unsigned short int WORD ;

/// Type BYTE 
typedef unsigned char BYTE ;

/// Type REAL (can be float/double/long double...)
//typedef long double REAL ;
typedef double REAL ;
//typedef float REAL ;
//typedef double REAL ;
typedef REAL* P_REAL ;
//@}

/** SVM type of float to use for caching
  * kernel evaluations. Using float saves
  * us some memory, but you can use double, too */
typedef float  CFLOAT;                  

/** the type used for storing feature ids */
typedef long FNUM;

/** the type used for storing feature values */
typedef float FVAL;  

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
#endif
