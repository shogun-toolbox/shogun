#ifndef _SVMLight_H___
#define _SVMLight_H___

#include "classifier/svm/SVM.h"
#include "kernel/Kernel.h"
#include "lib/common.h"
#include "classifier/svm/Optimizer.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include <float.h>

# define VERSION       "V3.50"
# define VERSION_DATE  "01.11.00"

# define DEF_PRECISION_LINEAR    1E-8
# define DEF_PRECISION_NONLINEAR 1E-14

class CSVMLight:public CSVM
{
 protected:
  typedef struct model {
    LONG    sv_num;	
    LONG    at_upper_bound;
    double  b;
    long*	supvec;
    double  *alpha;
    LONG    *index;       /* index from docnum to position in model */
    LONG    totdoc;       /* number of training documents */
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
    double svm_c;                /* upper bound C on alphas */
    double svm_costratio;        /* factor to multiply C for positive examples */
    double transduction_posratio;/* fraction of unlabeled examples to be */
    
    /* classified as positives */
    LONG   biased_hyperplane;    /* if nonzero, use hyperplane w*x+b=0 
				    otherwise w*x=0 */
    LONG   svm_maxqpsize;        /* size q of working set */
    LONG   svm_newvarsinqp;      /* new variables to enter the working set 
				    in each iteration */
    double epsilon_crit;         /* tolerable error for distances used 
				    in stopping criterion */
    double epsilon_shrink;       /* how much a multiplier should be above 
				    zero for shrinking */
    LONG   svm_iter_to_shrink;   /* iterations h after which an example can
				    be removed by shrinking */
    LONG   remove_inconsistent;  /* exclude examples with alpha at C and 
				    retrain */
    LONG   skip_final_opt_check; /* do not check KT-Conditions at the end of
				    optimization for examples removed by 
				    shrinking. WARNING: This might lead to 
				    sub-optimal solutions! */
    LONG   compute_loo;          /* if nonzero, computes leave-one-out
				    estimates */
    double rho;                  /* parameter in xi/alpha-estimates and for
				    pruning leave-one-out range [1..2] */
    LONG   xa_depth;             /* parameter in xi/alpha-estimates upper
				    bounding the number of SV the current
				    alpha_t is distributed over */
    CHAR predfile[200];          /* file for predicitions on unlabeled examples
				    in transduction */
    CHAR alphafile[200];         /* file to store optimal alphas in. use  
				    empty string if alphas should not be 
				    output */
    
    /* you probably do not want to touch the following */
    double epsilon_const;        /* tolerable error on eq-constraINT */
    double epsilon_a;            /* tolerable error on alphas at bounds */
    double opt_precision;        /* precision of solver, set to e.g. 1e-21 
				    if you get convergence problems */
    
    /* the following are only for internal use */
    LONG   svm_c_steps;          /* do so many steps for finding optimal C */
    double svm_c_factor;         /* increase C by this factor every step */
    double svm_costratio_unlab;
    double svm_unlabbound;
    double *svm_cost;            /* individual upper bounds for each var */
  } LEARN_PARM;
  
  typedef struct timing_profile {
    LONG   time_kernel;
    LONG   time_opti;
    LONG   time_shrink;
    LONG   time_update;
    LONG   time_model;
    LONG   time_check;
    LONG   time_select;
  } TIMING;
  
  typedef struct shrink_state {
    LONG   *active;
    LONG   *inactive_since;
    LONG   deactnum;
    double **a_history;
  } SHRINK_STATE;
  
 public:
  CSVMLight();
  virtual ~CSVMLight();
  
  virtual bool	train();

  double classify_example(LONG num);
  double model_length_s(MODEL*);
  void   clear_vector_n(double *, LONG);
  void   read_model(CHAR *, MODEL *, long, LONG);
  LONG   get_runtime();
  void   svm_learn();
  
  LONG optimize_to_convergence( LONG* docs, INT* label, LONG totdoc, 
					   SHRINK_STATE *shrink_state, MODEL *model, LONG *inconsistent, 
					   double *a, double *lin, TIMING *timing_profile, 
					   double *maxdiff, LONG heldout, LONG retrain);
  
  double compute_objective_function(double *, double *, INT *, LONG *);
  void   clear_index(LONG *);
  void   add_to_index(LONG *, LONG);
  LONG   compute_index(LONG *,long, LONG *);
  void optimize_svm(LONG* docs, INT* label, LONG *chosen, 
			       LONG *active2dnum, MODEL *model, LONG totdoc, 
			       LONG* working2dnum, LONG varnum, double *a, double* lin,
			       REAL *aicache, QP *qp, double *epsilon_crit_target);
  void   optimize_svm(LONG num, LONG *, LONG *, LONG *, LONG *, MODEL *, long, 
		      LONG *, long, double *, double *, LEARN_PARM *, REAL *, QP *, double *);
  void   compute_matrices_for_optimization(LONG *docs, INT *, LONG *, LONG *, 
					   LONG *, MODEL *, double *, 
					   double *, long, long, REAL *, QP *);
  LONG   calculate_svm_model(LONG *docs,INT *label,double *lin, double *a, double* a_old,LONG *working2dnum,MODEL *model);
  LONG   check_optimality(MODEL *model, INT *label, double *a, double* lin,
			  LONG totdoc, double *maxdiff, double epsilon_crit_org,
			  LONG *misclassified, LONG *inconsistent,LONG* active2dnum, 
			  LONG *last_suboptimal_at, LONG iteration) ;

  LONG   identify_inconsistent(double *, LONG *, LONG *, long, LEARN_PARM *, 
			       LONG *, LONG *);
  LONG   identify_misclassified(double *, INT *, long,
				MODEL *, LONG *, LONG *);
  LONG   identify_one_misclassified(double *, INT *, long,
				    MODEL *, LONG *, LONG *);
  LONG   incorporate_unlabeled_examples(MODEL *, LONG *,LONG *, LONG *,
					double *, double *, long, double *,
					LONG *, LONG *, long, LEARN_PARM *);
  void update_linear_component( LONG* docs, INT *label, 
					   LONG *active2dnum, double *a, double* a_old,
					   LONG *working2dnum, LONG totdoc,
					   double *lin, REAL *aicache, double *weights);
  LONG select_next_qp_subproblem_grad( INT *label, double *a,
						  double* lin, LONG totdoc, LONG qp_size, LONG *inconsistent, 
						  LONG* active2dnum, LONG* working2dnum, double *selcrit, 
						  LONG *select, LONG *key, LONG *chosen);
  LONG select_next_qp_subproblem_grad_cache(INT *label, double *a, 
						       double *lin, LONG totdoc, LONG qp_size, LONG *inconsistent, 
						       LONG* active2dnum, LONG* working2dnum, double *selcrit, 
						       LONG *select, LONG *key, LONG* chosen);
  
  void   select_top_n(double *, long, LONG *, LONG);
  void   init_shrink_state(SHRINK_STATE *, long, LONG);
  void   shrink_state_cleanup(SHRINK_STATE *);
  LONG   shrink_problem(LEARN_PARM *, SHRINK_STATE *, LONG *, LONG *, long,  
			long, long, double *, LONG *);
  void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				      double *lin,LONG totdoc,LONG iteration,
				      LEARN_PARM *learn_parm,LONG *inconsistent,
				      LONG *docs,MODEL *model,REAL *aicache,
				      double *weights, double* maxdiff) ;
  void compute_xa_estimates(MODEL *, LONG *, LONG *, long, LONG num, 
			    double *, double *, LEARN_PARM *, double *, double *, double *);
  double xa_estimate_error(MODEL *, LONG *, LONG *, long, LONG num, 
			   double *, double *, LEARN_PARM *);
  double xa_estimate_recall(MODEL *, LONG *, LONG *, long, LONG num, 
			    double *, double *, LEARN_PARM *);
  double xa_estimate_precision(MODEL *, LONG *, LONG *, long, LONG num, 
			       double *, double *, LEARN_PARM *);
  void avg_similarity_of_sv_of_one_class(MODEL *, LONG num, double *, LONG *, double *, double *);
  double most_similar_sv_of_same_class(MODEL *, LONG num, double *, long, LONG *, LEARN_PARM *);
  double distribute_alpha_t_greedily(LONG *, long, LONG num, double *, long, LONG *, LEARN_PARM *, double);
  double distribute_alpha_t_greedily_noindex(MODEL *, LONG num, double *, long, LONG *, LEARN_PARM *, double); 
  double estimate_margin_vcdim(MODEL *, double, double);
  double estimate_sphere(MODEL *);
  double estimate_r_delta_average(LONG num, LONG); 
  double estimate_r_delta(LONG num, LONG); 
  double length_of_longest_document_vector(LONG num, LONG); 
  
  void   write_model(FILE *, MODEL *);
  void   write_prediction(CHAR *, MODEL *, double *, double *, LONG *, LONG *, long, LEARN_PARM *);
   void   write_alphas(CHAR *, double *, INT *, LONG);
  
 protected:
  bool svm_loaded;
  MODEL* model;
  LEARN_PARM* learn_parm;
  LONG   verbosity;              /* verbosity level (0-4) */
  double* primal;
  double* dual;

  double init_margin;
  LONG   init_iter,precision_violations;
  double model_b;
  double opt_precision;
};

#endif
