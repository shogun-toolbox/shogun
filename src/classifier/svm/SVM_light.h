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

/** model */
struct MODEL {
/** sv num */
INT    sv_num;
/** at upper bound */
INT    at_upper_bound;
/** b */
double b;
/** supvec */
INT*   supvec;
/** alpha */
double *alpha;
/** index from docnum to position in model */
INT    *index;
/** number of training documents */
INT    totdoc;
/** kernel */
CKernel* kernel;

/* the following values are not written to file */
/** leave-one-out estimates */
double loo_error;
/** leave-one-out estimates */
double loo_recall;
/** leave-one-out estimates */
double loo_precision;

/** xi/alpha estimates */
double xa_error;
/** xi/alpha estimates */
double xa_recall;
/** xi/alpha estimates */
double xa_precision;
};

/** the type used for storing feature ids */
typedef INT FNUM;

/** the type used for storing feature values */
typedef double FVAL;  

/** learning parameters */
struct LEARN_PARM {
  /** selects between regression and classification */
  INT   type;
  /** upper bound C on alphas */
  double svm_c;
  /** regression epsilon (eps=1.0 for classification */
  double eps;
  /** factor to multiply C for positive examples */
  double svm_costratio;
  /** fraction of unlabeled examples to be */
  double transduction_posratio;
  /* classified as positives */
  /** if nonzero, use hyperplane w*x+b=0 otherwise w*x=0 */
  INT   biased_hyperplane;
  /** if nonzero, it will use the shared slack variable mode in
   * svm_learn_optimization. It requires that the slackid is set for every
   * training example
   */
  INT   sharedslack;
  /** size q of working set */
  INT   svm_maxqpsize;
  /** new variables to enter the working set in each iteration */
  INT   svm_newvarsinqp;
  /** size of kernel cache in megabytes */
  INT   kernel_cache_size;
  /** tolerable error for distances used in stopping criterion */
  double epsilon_crit;
  /** how much a multiplier should be above zero for shrinking */
  double epsilon_shrink;
  /** iterations h after which an example can be removed by shrinking */
  INT   svm_iter_to_shrink;
  /** number of iterations after which the optimizer terminates, if there was
   * no progress in maxdiff
   */
  INT   maxiter;
  /** exclude examples with alpha at C and retrain */
  INT   remove_inconsistent;
  /** do not check KT-Conditions at the end of optimization for examples
   * removed by shrinking. WARNING: This might lead to sub-optimal solutions!
   */
  INT   skip_final_opt_check;
  /** if nonzero, computes leave-one-out estimates */
  INT   compute_loo;
  /** parameter in xi/alpha-estimates and for pruning leave-one-out range
   * [1..2]
   */
  double rho;
  /** parameter in xi/alpha-estimates upper bounding the number of SV the
   * current alpha_t is distributed over
   */
  INT   xa_depth;
  /** file for predicitions on unlabeled examples in transduction */
  char predfile[200];
  /** file to store optimal alphas in. use empty string if alphas should not be
   * output
   */
  char alphafile[200];

  /* you probably do not want to touch the following */
  /** tolerable error on eq-constraint */
  double epsilon_const;
  /** tolerable error on alphas at bounds */
  double epsilon_a;
  /** precision of solver, set to e.g. 1e-21 if you get convergence problems */
  double opt_precision;

  /* the following are only for internal use */
  /** do so many steps for finding optimal C */
  INT   svm_c_steps;
  /** increase C by this factor every step */
  double svm_c_factor;
  /** costratio unlab */
  double svm_costratio_unlab;
  /** unlabbound */
  double svm_unlabbound;
  /** individual upper bounds for each var */
  double *svm_cost;
};

/** timing profile */
struct TIMING {
  /** time kernel */
  INT   time_kernel;
  /** time opti */
  INT   time_opti;
  /** time shrink */
  INT   time_shrink;
  /** time update */
  INT   time_update;
  /** time model */
  INT   time_model;
  /** time check */
  INT   time_check;
  /** time select */
  INT   time_select;
};


/** shrink state */
struct SHRINK_STATE
{
  /** active */
  INT   *active;
  /** inactive since */
  INT   *inactive_since;
  /** deactnum */
  INT   deactnum;
  /** for shrinking with non-linear kernel */
  double **a_history;
  /** maximum history */
  INT   maxhistory;
  /** for shrinking with linear kernel */
  double *last_a;
  /** for shrinking with linear kernel */
  double *last_lin;
};

/** class SVMlight */
class CSVMLight : public CSVM
{
 public:
  /** default constructor */
  CSVMLight();

  /** constructor
   *
   * @param C constant C
   * @param k kernel
   * @param lab labels
   */
  CSVMLight(DREAL C, CKernel* k, CLabels* lab);
  virtual ~CSVMLight();

  /** init SVM */
  void init();

  /** train SVM
   *
   * @return if training was successful
   */
  virtual bool train();

  /** setup AUC maximization
   *
   * @return if maximization was successful
   */
  bool setup_auc_maximization() ;

  /** get classifier type
   *
   * @return classifier type LIGHT
   */
  virtual inline EClassifierType get_classifier_type() { return CT_LIGHT; }

  /** get runtime
   *
   * @return runtime
   */
  INT   get_runtime();

  /** learn SVM */
  void   svm_learn();

  /** optimize to convergence
   *
   * @param docs the docs
   * @param label the label
   * @param totdoc the totdoc
   * @param shrink_state shrink state
   * @param inconsistent inconsistent
   * @param a a
   * @param lin lin
   * @param c c
   * @param timing_profile timing profile
   * @param maxdiff maximum diff
   * @param heldout held out
   * @param retrain retrain
   * @return something inty
   */
  INT optimize_to_convergence(INT* docs, INT* label, INT totdoc,
					   SHRINK_STATE *shrink_state, INT *inconsistent,
					   double *a, double *lin, double *c, TIMING *timing_profile,
					   double *maxdiff, INT heldout, INT retrain);

  /** compute objective function
   *
   * @param a a
   * @param lin lin
   * @param c c
   * @param eps epsilon
   * @param label label
   * @param totdoc totdoc
   * @return something floaty
   */
  virtual double compute_objective_function(double *a, double *lin, double *c, double eps, INT *label, INT totdoc);

  /** clear index
   *
   * @param index index
   */
  void   clear_index(INT *index);

  /** add to index
   *
   * @param index index
   * @param elem element at index
   */
  void   add_to_index(INT *index, INT elem);

  /** compute index
   *
   * @param binfeature binary feature
   * @param range range
   * @param index
   * @return something inty
   */
  INT   compute_index(INT *binfeature, INT range, INT *index);

  /** optimise SVM
   *
   * @param docs docs
   * @param label label
   * @param exclude_from_eq_const exclude from eq const
   * @param eq_target eq target
   * @param chosen chosen
   * @param active2dnum active 2D num
   * @param totdoc totdoc
   * @param working2dnum working 2D num
   * @param varnum var num
   * @param a a
   * @param lin lin
   * @param c c
   * @param aicache ai cache
   * @param qp QP
   * @param epsilon_crit_target epsilon crit target
   */
  void optimize_svm(INT* docs, INT* label,
		  INT *exclude_from_eq_const, double eq_target,
		  INT *chosen, INT *active2dnum,
		  INT totdoc, INT *working2dnum, INT varnum,
		  double *a, double *lin, double *c, DREAL *aicache, QP *qp,
		  double *epsilon_crit_target);

  /** compute matrices for optimization
   *
   * @param docs docs
   * @param label label
   * @param exclude_from_eq_const exclude from eq const
   * @param eq_target eq target
   * @param chosen chosen
   * @param active2dnum active 2D num
   * @param key key
   * @param a a
   * @param lin lin
   * @param c c
   * @param varnum var num
   * @param totdoc totdoc
   * @param aicache ai cache
   * @param qp QP
   */
  void compute_matrices_for_optimization(INT* docs, INT* label,
										 INT *exclude_from_eq_const, double eq_target,
										 INT *chosen, INT *active2dnum,
										 INT *key, double *a, double *lin, double *c,
										 INT varnum, INT totdoc, DREAL *aicache, QP *qp);

  /** compute matrices for optimization in parallel
   *
   * @param docs docs
   * @param label label
   * @param exclude_from_eq_const exclude from eq const
   * @param eq_target eq target
   * @param chosen chosen
   * @param active2dnum active 2D num
   * @param key key
   * @param a a
   * @param lin lin
   * @param c c
   * @param varnum var num
   * @param totdoc totdoc
   * @param aicache ai cache
   * @param qp QP
   */
  void compute_matrices_for_optimization_parallel(INT* docs, INT* label,
												  INT *exclude_from_eq_const, double eq_target,
												  INT *chosen, INT *active2dnum,
												  INT *key, double *a, double *lin, double *c,
												  INT varnum, INT totdoc, DREAL *aicache, QP *qp);

  /** calculate SVM model
   *
   * @param docs docs
   * @param label label
   * @param lin lin
   * @param a a
   * @param a_old old a
   * @param c c
   * @param working2dnum working 2D num
   * @param active2dnum active 2D num
   * @return something inty
   */
  INT   calculate_svm_model(INT* docs, INT *label,double *lin, double *a, double* a_old, double *c, INT *working2dnum, INT *active2dnum);

  /** check optimality
   *
   * @param label label
   * @param a a
   * @param lin lin
   * @param c c
   * @param totdoc totdoc
   * @param maxdiff maximum diff
   * @param epsilon_crit_org epsilon crit org
   * @param misclassified misclassified
   * @param inconsistent inconsistent
   * @param active2dnum active 2D num
   * @param last_suboptimal_at last suboptimal at
   * @param iteration iteration
   * @return something inty
   */
  INT   check_optimality(INT *label, double *a, double* lin, double *c,
			  INT totdoc, double *maxdiff, double epsilon_crit_org,
			  INT *misclassified, INT *inconsistent,INT* active2dnum,
			  INT *last_suboptimal_at, INT iteration);

  /** update linear component
   *
   * @param docs docs
   * @param label label
   * @param active2dnum active 2D num
   * @param a a
   * @param a_old old a
   * @param working2dnum working 2D num
   * @param totdoc totdoc
   * @param lin lin
   * @param aicache ai cache
   * @param c c
   */
  virtual void update_linear_component(INT* docs, INT *label,
							   INT *active2dnum, double *a, double* a_old,
							   INT *working2dnum, INT totdoc,
							   double *lin, DREAL *aicache, double* c);

  // MKL stuff
  /** update linear component MKL
   *
   * @param docs docs
   * @param label label
   * @param active2dnum active 2D num
   * @param a a
   * @param a_old old a
   * @param working2dnum working 2D num
   * @param totdoc totdoc
   * @param lin lin
   * @param aicache ai cache
   */
  void update_linear_component_mkl(INT* docs, INT *label,
								   INT *active2dnum, double *a, double* a_old,
								   INT *working2dnum, INT totdoc,
								   double *lin, DREAL *aicache);

  /** update linear component MKL
   *
   * @param docs docs
   * @param label label
   * @param active2dnum active 2D num
   * @param a a
   * @param a_old old a
   * @param working2dnum working 2D num
   * @param totdoc totdoc
   * @param lin lin
   * @param aicache ai cache
   */
  void update_linear_component_mkl_linadd(INT* docs, INT *label,
										  INT *active2dnum, double *a, double* a_old,
										  INT *working2dnum, INT totdoc,
										  double *lin, DREAL *aicache);

  /** select next qp subproblem grad
   *
   * @param label label
   * @param a a
   * @param lin lin
   * @param c c
   * @param totdoc totdoc
   * @param qp_size size of qp
   * @param inconsistent inconsistent
   * @param active2dnum active 2D num
   * @param working2dnum working 2D num
   * @param selcrit selcrit
   * @param select select
   * @param cache_only cache only
   * @param key key
   * @param chosen chosen
   * @return something inty
   */
  INT select_next_qp_subproblem_grad( INT *label, double *a,
						  double* lin, double* c, INT totdoc, INT qp_size, INT *inconsistent,
						  INT* active2dnum, INT* working2dnum, double *selcrit,
						  INT *select, INT cache_only, INT *key, INT *chosen);

  /** select next qp subproblem rand
   *
   * @param label label
   * @param a a
   * @param lin lin
   * @param c c
   * @param totdoc totdoc
   * @param qp_size size of qp
   * @param inconsistent inconsistent
   * @param active2dnum active 2D num
   * @param working2dnum working 2D num
   * @param selcrit selcrit
   * @param select select
   * @param key key
   * @param chosen chosen
   * @param iteration iteration
   * @return something inty
   */
  INT select_next_qp_subproblem_rand(INT* label, double *a, double *lin,
				    double *c, INT totdoc, INT qp_size,
				    INT *inconsistent, INT *active2dnum, INT *working2dnum,
				    double *selcrit, INT *select, INT *key,
					INT *chosen,
				    INT iteration);

  /** select top n
   *
   * @param selcrit selcrit
   * @param range range
   * @param select select
   * @param n n
   */
  void   select_top_n(double *selcrit, INT range, INT *select, INT n);

  /** init shrink state
   *
   * @param shrink_state shrink state
   * @param totdoc totdoc
   * @param maxhistory maximum history
   */
  void   init_shrink_state(SHRINK_STATE *shrink_state, INT totdoc, INT maxhistory);

  /** cleanup shrink state
   *
   * @param shrink_state shrink state
   */
  void   shrink_state_cleanup(SHRINK_STATE *shrink_state);

  /** shrink problem
   *
   * @param shrink_state shrink state
   * @param active2dnum active 2D num
   * @param last_suboptimal_at last suboptimal at
   * @param iteration iteration
   * @param totdoc totdoc
   * @param minshrink minimal shrink
   * @param a a
   * @param inconsistent inconsistent
   * @param c c
   * @param lin lin
   * @param label label
   * @return something inty
   */
  INT shrink_problem(SHRINK_STATE *shrink_state, INT *active2dnum, INT *last_suboptimal_at,
		    INT iteration, INT totdoc, INT minshrink,
		    double *a, INT *inconsistent, double* c, double* lin, int* label);

  /** reactivate inactive examples
   *
   * @param label label
   * @param a a
   * @param shrink_state shrink state
   * @param lin lin
   * @param c c
   * @param totdoc totdoc
   * @param iteration iteration
   * @param inconsistent inconsistent
   * @param docs docs
   * @param aicache ai cache
   * @param maxdiff maximum diff
   */
  virtual void   reactivate_inactive_examples(INT *label,double *a,SHRINK_STATE *shrink_state,
				      double *lin, double *c, INT totdoc,INT iteration,
				      INT *inconsistent,
				      INT *docs,DREAL *aicache,
				      double* maxdiff);

protected:
   /** compute kernel
	*
	* @param i at index i
	* @param j at index j
	* @return computed kernel item at index i, j
	*/
   inline virtual DREAL compute_kernel(INT i, INT j)
	   {
		   if (use_precomputed_subkernels)
		   {
			   if (j>i)
				   CMath::swap(i,j) ;
			   DREAL sum=0 ;
			   INT num_weights=-1 ;
			   const DREAL * w = kernel->get_subkernel_weights(num_weights) ;
			   for (INT n=0; n<num_precomputed_subkernels; n++)
				   if (w[n]!=0)
					   sum += w[n]*precomputed_subkernels[n][i*(i+1)/2+j] ;
			   return sum ;
		   }
		   else
			   return kernel->kernel(i, j) ;
	   }

	/** helper for compute kernel
	 *
	 * @param p p
	 */
	static void* compute_kernel_helper(void* p);

	/** helper for update linear component linadd
	 *
	 * @param p p
	 */
	static void* update_linear_component_linadd_helper(void* p);

	/** helper for update linear component MKL linadd
	 *
	 * @param p p
	 */
	static void* update_linear_component_mkl_linadd_helper(void* p);

	/** helper for reactivate inactive examples vanilla
	 *
	 * @param p p
	 */
	static void* reactivate_inactive_examples_vanilla_helper(void* p);

	/** helper for reactivate inactive examples linadd
	 *
	 * @param p p
	 */
	static void* reactivate_inactive_examples_linadd_helper(void* p);

#ifdef USE_CPLEX
	/** init cplex
	 *
	 * @return if init was successful
	 */
	bool init_cplex();

	/** cleanup cplex
	 *
	 * @return if cleanup was successful
	 */
	bool cleanup_cplex();
#endif
   
 protected:
  /** model */
  MODEL* model;
  /** learn parameters */
  LEARN_PARM* learn_parm;
  /** verbosity level (0-4) */
  INT   verbosity;

  /** init margin */
  double init_margin;
  /** init iter */
  INT   init_iter;
  /** precision violations */
  INT precision_violations;
  /** model b */
  double model_b;
  /** opt precision */
  double opt_precision;

  // MKL stuff

  /** Matrix that stores the contribution by each kernel for each example (for
   * current alphas)
   */
  DREAL* W;
  /** current margin */
  DREAL rho;
  /** current relative w gap */
  DREAL w_gap;
  /** regularization parameter for w smoothing */
  DREAL lp_C;
  /** number of iteration */
  INT count;
  /** current alpha gap */
  DREAL mymaxdiff;
  /** number of alpha constraint rows */
  INT num_rows;
  /** number of active alpha constraint rows */
  INT num_active_rows;
  /** a buffer of length num */
  DREAL *buffer_num;
  /** a buffer of length num_cols */
  DREAL *buffer_numcols;
  // MKL kernel precomputation
  /** precomputed subkernels */
  SHORTREAL ** precomputed_subkernels;
  /** number of precomputed subkernels */
  INT num_precomputed_subkernels;
  /** if kernel cache is used */
  bool use_kernel_cache;

#ifdef USE_CPLEX
  /** env */
  CPXENVptr     env;
  /** lp */
  CPXLPptr      lp;
  /** if lp is initialized */
  bool          lp_initialized ;
#endif

};
#endif //USE_SVMLIGHT
#endif
