#ifndef _SVMLight_H___
#define _SVMLight_H___

#include "classifier/svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "classifier/svm/Optimizer.h"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include <float.h>

# define VERSION       "V3.50"
# define VERSION_DATE  "01.11.00"

# define DEF_PRECISION_LINEAR    1E-8
# define DEF_PRECISION_NONLINEAR 1E-14
# define MAXSHRINK 50000

class CSVMLight:public CSVM
{
 protected:
  typedef struct model {
    long int    sv_num;	
    long int    at_upper_bound;
    double  b;
    long*	supvec;
    double  *alpha;
    long int    *index;       /* index from docnum to position in model */
    long int    totdoc;       /* number of training documents */
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
	  long   type;                 /* selects between regression and
									  classification */
	  double svm_c;                /* upper bound C on alphas */
	  double eps;                  /* regression epsilon (eps=1.0 for
									  classification */
	  double svm_costratio;        /* factor to multiply C for positive examples */
	  double transduction_posratio;/* fraction of unlabeled examples to be */
	  /* classified as positives */
	  long   biased_hyperplane;    /* if nonzero, use hyperplane w*x+b=0 
									  otherwise w*x=0 */
	  long   sharedslack;          /* if nonzero, it will use the shared
									  slack variable mode in
									  svm_learn_optimization. It requires
									  that the slackid is set for every
									  training example */
	  long   svm_maxqpsize;        /* size q of working set */
	  long   svm_newvarsinqp;      /* new variables to enter the working set 
									  in each iteration */
	  long   kernel_cache_size;    /* size of kernel cache in megabytes */
	  double epsilon_crit;         /* tolerable error for distances used 
									  in stopping criterion */
	  double epsilon_shrink;       /* how much a multiplier should be above 
									  zero for shrinking */
	  long   svm_iter_to_shrink;   /* iterations h after which an example can
									  be removed by shrinking */
	  long   maxiter;              /* number of iterations after which the
									  optimizer terminates, if there was
									  no progress in maxdiff */
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
  } LEARN_PARM;

  typedef struct timing_profile {
	  long int   time_kernel;
	  long int   time_opti;
	  long int   time_shrink;
	  long int   time_update;
	  long int   time_model;
	  long int   time_check;
	  long int   time_select;
  } TIMING;
  

typedef struct shrink_state {                                              
  long   *active;                                      
  long   *inactive_since;                                
  long   deactnum;                          
  double **a_history;  /* for shrinking with non-linear kernel */    
  long   maxhistory;           
  double *last_a;      /* for shrinking with linear kernel */             
  double *last_lin;    /* for shrinking with linear kernel */
} SHRINK_STATE;                                                         

 public:
  CSVMLight();
  virtual ~CSVMLight();
  
  virtual bool	train();

  double classify_example(long int num);
  double model_length_s(MODEL*);
  void   clear_vector_n(double *, long int);
  void   read_model(CHAR *, MODEL *, long, long int);
  long int   get_runtime();
  void   svm_learn();
  
  long int optimize_to_convergence(long int* docs, INT* label, long int totdoc, 
					   SHRINK_STATE *shrink_state, MODEL *model, long int *inconsistent, 
					   double *a, double *lin, double *c, TIMING *timing_profile, 
					   double *maxdiff, long int heldout, long int retrain);
  
  double compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, long int *active2dnum);
  void   clear_index(long int *);
  void   add_to_index(long int *, long int);
  long int   compute_index(long int *,long, long int *);

  void optimize_svm(LONG* docs, INT* label,
		  long int *exclude_from_eq_const, double eq_target,
		  long int *chosen, long int *active2dnum, MODEL *model, 
		  long int totdoc, long int *working2dnum, long int varnum, 
		  double *a, double *lin, double *c, REAL *aicache, QP *qp, 
		  double *epsilon_crit_target);

  void compute_matrices_for_optimization(LONG* docs, INT* label, 
		  long *exclude_from_eq_const, double eq_target,
		  long int *chosen, long int *active2dnum, 
		  long int *key, MODEL *model, double *a, double *lin, double *c, 
		  long int varnum, long int totdoc, REAL *aicache, QP *qp);

  long int   calculate_svm_model(LONG* docs, INT *label,double *lin, double *a, double* a_old, double *c, long int *working2dnum, long int *active2dnum, MODEL *model);
  long int   check_optimality(MODEL *model, INT *label, double *a, double* lin, double *c,
			  long int totdoc, double *maxdiff, double epsilon_crit_org,
			  long int *misclassified, long int *inconsistent,long int* active2dnum, 
			  long int *last_suboptimal_at, long int iteration) ;

  long int   identify_inconsistent(double *, long int *, long int *, long, LEARN_PARM *, 
			       long int *, long int *);
  long int   identify_misclassified(double *, INT *, long,
				MODEL *, long int *, long int *);
  long int   identify_one_misclassified(double *, INT *, long,
				    MODEL *, long int *, long int *);
  long int   incorporate_unlabeled_examples(MODEL *, long int *,long int *, long int *,
					double *, double *, long, double *,
					long int *, long int *, long, LEARN_PARM *);
  void update_linear_component(LONG* docs, INT *label, 
					   long int *active2dnum, double *a, double* a_old,
					   long int *working2dnum, long int totdoc,
					   double *lin, REAL *aicache);
  long int select_next_qp_subproblem_grad( INT *label, double *a,
						  double* lin, double* c, long int totdoc, long int qp_size, long int *inconsistent, 
						  long int* active2dnum, long int* working2dnum, double *selcrit, 
						  long int *select, long int cache_only, long int *key, long int *chosen);
  long select_next_qp_subproblem_rand(INT* label, double *a, double *lin, 
				    double *c, long int totdoc, long int qp_size, 
				    long int *inconsistent, long int *active2dnum, long int *working2dnum, 
				    double *selcrit, long int *select, long int *key, 
					long int *chosen, 
				    long int iteration);
  
  void   select_top_n(double *, long, long int *, long int);
  void   init_shrink_state(SHRINK_STATE *, long, long int);
  void   shrink_state_cleanup(SHRINK_STATE *);
  long shrink_problem(SHRINK_STATE *shrink_state, long int *active2dnum, long int *last_suboptimal_at, 
		    long int iteration, long int totdoc, long int minshrink, 
		    double *a, long int *inconsistent);
  void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				      double *lin, double *c, long int totdoc,long int iteration,
				      long int *inconsistent,
				      long int *docs,MODEL *model,REAL *aicache,
				      double* maxdiff) ;
  void compute_xa_estimates(MODEL *, long int *, long int *, long, long int num, 
			    double *, double *, LEARN_PARM *, double *, double *, double *);
  double xa_estimate_error(MODEL *, long int *, long int *, long, long int num, 
			   double *, double *, LEARN_PARM *);
  double xa_estimate_recall(MODEL *, long int *, long int *, long, long int num, 
			    double *, double *, LEARN_PARM *);
  double xa_estimate_precision(MODEL *, long int *, long int *, long, long int num, 
			       double *, double *, LEARN_PARM *);
  void avg_similarity_of_sv_of_one_class(MODEL *, long int num, double *, long int *, double *, double *);
  double most_similar_sv_of_same_class(MODEL *, long int num, double *, long, long int *, LEARN_PARM *);
  double distribute_alpha_t_greedily(long int *, long, long int num, double *, long, long int *, LEARN_PARM *, double);
  double distribute_alpha_t_greedily_noindex(MODEL *, long int num, double *, long, long int *, LEARN_PARM *, double); 
  double estimate_margin_vcdim(MODEL *, double, double);
  double estimate_sphere(MODEL *);
  double estimate_r_delta_average(long int num, long int); 
  double estimate_r_delta(long int num, long int); 
  double length_of_longest_document_vector(long int num, long int); 
  
  void   write_model(FILE *, MODEL *);
  void   write_prediction(CHAR *, MODEL *, double *, double *, long int *, long int *, long, LEARN_PARM *);
   void   write_alphas(CHAR *, double *, INT *, long int);
  
 protected:
  bool svm_loaded;
  MODEL* model;
  LEARN_PARM* learn_parm;
  long int   verbosity;              /* verbosity level (0-4) */
  double* primal;
  double* dual;

  double init_margin;
  long int   init_iter,precision_violations;
  double model_b;
  double opt_precision;
  REAL* W;
  REAL rho ;
  REAL *rhos ;
  REAL sumabsgammas ;
  REAL w_gap ;
};

#endif
