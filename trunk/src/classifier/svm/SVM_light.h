/***********************************************************************/
/*                                                                     */
/*   SVM_light.h                                                       */
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
/*   THIS INCLUDES THE FOLLOWING ADDITIONS                             */
/*   Generic Kernel Interfacing: Soeren Sonnenburg                     */
/*   Parallizations: Soeren Sonnenburg                                 */
/*   Multiple Kernel Learning: Gunnar Raetsch, Soeren Sonnenburg       */
/*   Linadd Speedup: Gunnar Raetsch, Soeren Sonnenburg                 */
/*                                                                     */
/***********************************************************************/
#ifndef _SVMLight_H___
#define _SVMLight_H___

#include "lib/config.h"

#ifdef USE_SVMLIGHT
#include "classifier/svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/Mathematics.h"
#include "lib/common.h"
#include "classifier/svm/Optimizer.h"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}
#endif

# define VERSION       "V3.50 -- correct??"
# define VERSION_DATE  "01.11.00 -- correct??"

# define MAXSHRINK 50000

typedef struct model {
INT    sv_num;	
INT    at_upper_bound;
double  b;
INT*	supvec;
double  *alpha;
INT    *index;       /* index from docnum to position in model */
INT    totdoc;       /* number of training documents */
CKernel* kernel; /* kernel */

/* the following values are not written to file */
double  loo_error,loo_recall,loo_precision; /* leave-one-out estimates */
double  xa_error,xa_recall,xa_precision;    /* xi/alpha estimates */
} MODEL;

/** the type used for storing feature ids */
typedef INT FNUM;

/** the type used for storing feature values */
typedef double FVAL;  

typedef struct learn_parm {
  INT   type;                 /* selects between regression and
								  classification */
  double svm_c;                /* upper bound C on alphas */
  double eps;                  /* regression epsilon (eps=1.0 for
								  classification */
  double svm_costratio;        /* factor to multiply C for positive examples */
  double transduction_posratio;/* fraction of unlabeled examples to be */
  /* classified as positives */
  INT   biased_hyperplane;    /* if nonzero, use hyperplane w*x+b=0 
								  otherwise w*x=0 */
  INT   sharedslack;          /* if nonzero, it will use the shared
								  slack variable mode in
								  svm_learn_optimization. It requires
								  that the slackid is set for every
								  training example */
  INT   svm_maxqpsize;        /* size q of working set */
  INT   svm_newvarsinqp;      /* new variables to enter the working set 
								  in each iteration */
  INT   kernel_cache_size;    /* size of kernel cache in megabytes */
  double epsilon_crit;         /* tolerable error for distances used 
								  in stopping criterion */
  double epsilon_shrink;       /* how much a multiplier should be above 
								  zero for shrinking */
  INT   svm_iter_to_shrink;   /* iterations h after which an example can
								  be removed by shrinking */
  INT   maxiter;              /* number of iterations after which the
								  optimizer terminates, if there was
								  no progress in maxdiff */
  INT   remove_inconsistent;  /* exclude examples with alpha at C and 
								  retrain */
  INT   skip_final_opt_check; /* do not check KT-Conditions at the end of
								  optimization for examples removed by 
								  shrinking. WARNING: This might lead to 
								  sub-optimal solutions! */
  INT   compute_loo;          /* if nonzero, computes leave-one-out
								  estimates */
  double rho;                  /* parameter in xi/alpha-estimates and for
								  pruning leave-one-out range [1..2] */
  INT   xa_depth;             /* parameter in xi/alpha-estimates upper
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
  INT   svm_c_steps;          /* do so many steps for finding optimal C */
  double svm_c_factor;         /* increase C by this factor every step */
  double svm_costratio_unlab;
  double svm_unlabbound;
  double *svm_cost;            /* individual upper bounds for each var */
} LEARN_PARM;

typedef struct timing_profile {
  INT   time_kernel;
  INT   time_opti;
  INT   time_shrink;
  INT   time_update;
  INT   time_model;
  INT   time_check;
  INT   time_select;
} TIMING;
  

typedef struct shrink_state {                                              
  INT   *active;                                      
  INT   *inactive_since;                                
  INT   deactnum;                          
  double **a_history;  /* for shrinking with non-linear kernel */    
  INT   maxhistory;           
  double *last_a;      /* for shrinking with linear kernel */             
  double *last_lin;    /* for shrinking with linear kernel */
} SHRINK_STATE;                                                         

class CSVMLight:public CSVM
{
 public:
  CSVMLight();
  CSVMLight(DREAL C, CKernel* k, CLabels* lab);
  virtual ~CSVMLight();
  
  void init();
  virtual bool	train();
  bool setup_auc_maximization() ;
  inline EClassifierType get_classifier_type() { return CT_LIGHT; }

  INT   get_runtime();
  void   svm_learn();
  
  INT optimize_to_convergence(INT* docs, INT* label, INT totdoc, 
					   SHRINK_STATE *shrink_state, MODEL *model, INT *inconsistent, 
					   double *a, double *lin, double *c, TIMING *timing_profile, 
					   double *maxdiff, INT heldout, INT retrain);
  
  virtual double compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, INT totdoc);

  void   clear_index(INT *);
  void   add_to_index(INT *, INT);
  INT   compute_index(INT *,INT, INT *);

  void optimize_svm(INT* docs, INT* label,
		  INT *exclude_from_eq_const, double eq_target,
		  INT *chosen, INT *active2dnum, MODEL *model, 
		  INT totdoc, INT *working2dnum, INT varnum, 
		  double *a, double *lin, double *c, DREAL *aicache, QP *qp, 
		  double *epsilon_crit_target);

  void compute_matrices_for_optimization(INT* docs, INT* label, 
										 INT *exclude_from_eq_const, double eq_target,
										 INT *chosen, INT *active2dnum, 
										 INT *key, MODEL *model, double *a, double *lin, double *c, 
										 INT varnum, INT totdoc, DREAL *aicache, QP *qp);
  void compute_matrices_for_optimization_parallel(INT* docs, INT* label, 
												  INT *exclude_from_eq_const, double eq_target,
												  INT *chosen, INT *active2dnum, 
												  INT *key, MODEL *model, double *a, double *lin, double *c, 
												  INT varnum, INT totdoc, DREAL *aicache, QP *qp);
  
  INT   calculate_svm_model(INT* docs, INT *label,double *lin, double *a, double* a_old, double *c, INT *working2dnum, INT *active2dnum, MODEL *model);
  INT   check_optimality(MODEL *model, INT *label, double *a, double* lin, double *c,
			  INT totdoc, double *maxdiff, double epsilon_crit_org,
			  INT *misclassified, INT *inconsistent,INT* active2dnum, 
			  INT *last_suboptimal_at, INT iteration) ;

  virtual void update_linear_component(INT* docs, INT *label, 
							   INT *active2dnum, double *a, double* a_old,
							   INT *working2dnum, INT totdoc,
							   double *lin, DREAL *aicache, double* c);
  // MKL stuff
  void update_linear_component_mkl(INT* docs, INT *label, 
								   INT *active2dnum, double *a, double* a_old,
								   INT *working2dnum, INT totdoc,
								   double *lin, DREAL *aicache);
  void update_linear_component_mkl_linadd(INT* docs, INT *label, 
										  INT *active2dnum, double *a, double* a_old,
										  INT *working2dnum, INT totdoc,
										  double *lin, DREAL *aicache);
  
  INT select_next_qp_subproblem_grad( INT *label, double *a,
						  double* lin, double* c, INT totdoc, INT qp_size, INT *inconsistent, 
						  INT* active2dnum, INT* working2dnum, double *selcrit, 
						  INT *select, INT cache_only, INT *key, INT *chosen);
  INT select_next_qp_subproblem_rand(INT* label, double *a, double *lin, 
				    double *c, INT totdoc, INT qp_size, 
				    INT *inconsistent, INT *active2dnum, INT *working2dnum, 
				    double *selcrit, INT *select, INT *key, 
					INT *chosen, 
				    INT iteration);
  
  void   select_top_n(double *, INT, INT *, INT);
  void   init_shrink_state(SHRINK_STATE *, INT, INT);
  void   shrink_state_cleanup(SHRINK_STATE *);
  INT shrink_problem(SHRINK_STATE *shrink_state, INT *active2dnum, INT *last_suboptimal_at, 
		    INT iteration, INT totdoc, INT minshrink, 
		    double *a, INT *inconsistent, double* c, double* lin, int* label);
  virtual void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				      double *lin, double *c, INT totdoc,INT iteration,
				      INT *inconsistent,
				      INT *docs,MODEL *model,DREAL *aicache,
				      double* maxdiff) ;
protected:
   inline virtual DREAL compute_kernel(INT i, INT j)
	   {
		   if (use_precomputed_subkernels)
		   {
			   if (j>i)
				   CMath::swap(i,j) ;
			   DREAL sum=0 ;
			   INT num_weights=-1 ;
			   //INT num = get_kernel()->get_rhs()->get_num_vectors() ;
			   const DREAL * w = CKernelMachine::get_kernel()->get_subkernel_weights(num_weights) ;
			   for (INT n=0; n<num_precomputed_subkernels; n++)
				   if (w[n]!=0)
					   sum += w[n]*precomputed_subkernels[n][i*(i+1)/2+j] ;
			   return sum ;
		   }
		   else
			   return CKernelMachine::get_kernel()->kernel(i, j) ;
	   }
	static void* compute_kernel_helper(void* p);
	static void* update_linear_component_linadd_helper(void* p);
	static void* update_linear_component_mkl_linadd_helper(void* p);
	static void* reactivate_inactive_examples_vanilla_helper(void* p);
	static void* reactivate_inactive_examples_linadd_helper(void* p);

#ifdef USE_CPLEX
	bool init_cplex();
	bool cleanup_cplex();
#endif
   
 protected:
  bool svm_loaded;
  MODEL* model;
  LEARN_PARM* learn_parm;
  INT   verbosity;              /* verbosity level (0-4) */

  double init_margin;
  INT   init_iter,precision_violations;
  double model_b;
  double opt_precision;
  
  // MKL stuff
  DREAL* W;             // Matrix that stores the contribution by each kernel 
                       // for each example (for current alphas)
  DREAL rho ;           // current margin
  DREAL w_gap ;         // current relative w gap
  DREAL lp_C ;          // regularization parameter for w smoothing
  INT count ;          // number of iteration
  DREAL mymaxdiff ;     // current alpha gap
  INT num_rows ;       // number of alpha constraint rows 
  INT num_active_rows ;// number of active alpha constraint rows
  DREAL *buffer_num ;   // a buffer of length num
  DREAL *buffer_numcols ;   // a buffer of length num_cols
  // MKL kernel precomputation
  SHORTREAL ** precomputed_subkernels ;
  INT num_precomputed_subkernels ;
  bool use_kernel_cache ;

#ifdef USE_CPLEX
  CPXENVptr     env ;
  CPXLPptr      lp ;
  bool          lp_initialized ;
#endif

};
#endif //USE_SVMLIGHT
#endif
