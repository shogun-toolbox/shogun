/***********************************************************************/
/*                                                                     */
/*   SVMLight.h                                                        */
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

#include <shogun/lib/config.h>

#ifdef USE_SVMLIGHT
#include <shogun/classifier/svm/SVM.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

namespace shogun
{
//# define VERSION       "V3.50 -- correct??"
//# define VERSION_DATE  "01.11.00 -- correct??"

# define DEF_PRECISION 1E-14
# define MAXSHRINK 50000

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** model */
struct MODEL {
/** sv num */
int32_t    sv_num;
/** at upper bound */
int32_t    at_upper_bound;
/** b */
float64_t b;
/** supvec */
int32_t*   supvec;
/** alpha */
float64_t *alpha;
/** index from docnum to position in model */
int32_t    *index;
/** number of training documents */
int32_t    totdoc;
/** kernel */
CKernel* kernel;

/* the following values are not written to file */
/** leave-one-out estimates */
float64_t loo_error;
/** leave-one-out estimates */
float64_t loo_recall;
/** leave-one-out estimates */
float64_t loo_precision;

/** xi/alpha estimates */
float64_t xa_error;
/** xi/alpha estimates */
float64_t xa_recall;
/** xi/alpha estimates */
float64_t xa_precision;
};

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

/** the type used for storing feature ids */
typedef int32_t FNUM;

/** the type used for storing feature values */
typedef float64_t FVAL;

/** learning parameters */
struct LEARN_PARM {
  /** selects between regression and classification */
  int32_t   type;
  /** upper bound C on alphas */
  float64_t svm_c;
  /** regression epsilon (eps=-1.0 for classification */
  float64_t* eps;
  /** factor to multiply C for positive examples */
  float64_t svm_costratio;
  /** fraction of unlabeled examples to be */
  float64_t transduction_posratio;
  /* classified as positives */
  /** if nonzero, use hyperplane w*x+b=0 otherwise w*x=0 */
  int32_t   biased_hyperplane;
  /** if nonzero, it will use the shared slack variable mode in
   * svm_learn_optimization. It requires that the slackid is set for every
   * training example
   */
  int32_t   sharedslack;
  /** size q of working set */
  int32_t   svm_maxqpsize;
  /** new variables to enter the working set in each iteration */
  int32_t   svm_newvarsinqp;
  /** size of kernel cache in megabytes */
  int32_t   kernel_cache_size;
  /** tolerable error for distances used in stopping criterion */
  float64_t epsilon_crit;
  /** how much a multiplier should be above zero for shrinking */
  float64_t epsilon_shrink;
  /** iterations h after which an example can be removed by shrinking */
  int32_t   svm_iter_to_shrink;
  /** number of iterations after which the optimizer terminates, if there was
   * no progress in maxdiff
   */
  int32_t   maxiter;
  /** exclude examples with alpha at C and retrain */
  int32_t   remove_inconsistent;
  /** do not check KT-Conditions at the end of optimization for examples
   * removed by shrinking. WARNING: This might lead to sub-optimal solutions!
   */
  int32_t   skip_final_opt_check;
  /** if nonzero, computes leave-one-out estimates */
  int32_t   compute_loo;
  /** parameter in xi/alpha-estimates and for pruning leave-one-out range
   * [1..2]
   */
  float64_t rho;
  /** parameter in xi/alpha-estimates upper bounding the number of SV the
   * current alpha_t is distributed over
   */
  int32_t   xa_depth;
  /** file for predicitions on unlabeled examples in transduction */
  char predfile[200];
  /** file to store optimal alphas in. use empty string if alphas should not be
   * output
   */
  char alphafile[200];

  /* you probably do not want to touch the following */
  /** tolerable error on eq-constraint */
  float64_t epsilon_const;
  /** tolerable error on alphas at bounds */
  float64_t epsilon_a;
  /** precision of solver, set to e.g. 1e-21 if you get convergence problems */
  float64_t opt_precision;

  /* the following are only for internal use */
  /** do so many steps for finding optimal C */
  int32_t   svm_c_steps;
  /** increase C by this factor every step */
  float64_t svm_c_factor;
  /** costratio unlab */
  float64_t svm_costratio_unlab;
  /** unlabbound */
  float64_t svm_unlabbound;
  /** individual upper bounds for each var */
  float64_t *svm_cost;
};

/** timing profile */
struct TIMING {
  /** time kernel */
  int32_t   time_kernel;
  /** time opti */
  int32_t   time_opti;
  /** time shrink */
  int32_t   time_shrink;
  /** time update */
  int32_t   time_update;
  /** time model */
  int32_t   time_model;
  /** time check */
  int32_t   time_check;
  /** time select */
  int32_t   time_select;
};


/** shrink state */
struct SHRINK_STATE
{
  /** active */
  int32_t   *active;
  /** inactive since */
  int32_t   *inactive_since;
  /** deactnum */
  int32_t   deactnum;
  /** for shrinking with non-linear kernel */
  float64_t **a_history;
  /** maximum history */
  int32_t   maxhistory;
  /** for shrinking with linear kernel */
  float64_t *last_a;
  /** for shrinking with linear kernel */
  float64_t *last_lin;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief class SVMlight */
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
  CSVMLight(float64_t C, CKernel* k, CLabels* lab);
  virtual ~CSVMLight();

  /** init SVM */
  void init();

  /** get classifier type
   *
   * @return classifier type LIGHT
   */
  virtual EMachineType get_classifier_type() { return CT_LIGHT; }

  /** get runtime
   *
   * @return runtime
   */
  int32_t   get_runtime();


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
  int32_t optimize_to_convergence(
	int32_t* docs, int32_t* label, int32_t totdoc, SHRINK_STATE *shrink_state,
	int32_t *inconsistent, float64_t *a, float64_t *lin, float64_t *c,
	TIMING *timing_profile, float64_t *maxdiff, int32_t heldout,
	int32_t retrain);

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
  virtual float64_t compute_objective_function(
	float64_t *a, float64_t *lin, float64_t *c, float64_t* eps, int32_t *label,
	int32_t totdoc);

  /** clear index
   *
   * @param index index
   */
  void   clear_index(int32_t *index);

  /** add to index
   *
   * @param index index
   * @param elem element at index
   */
  void   add_to_index(int32_t *index, int32_t elem);

  /** compute index
   *
   * @param binfeature binary feature
   * @param range range
   * @param index
   * @return something inty
   */
  int32_t   compute_index(int32_t *binfeature, int32_t range, int32_t *index);

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
  void optimize_svm(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t totdoc,
	int32_t *working2dnum, int32_t varnum, float64_t *a, float64_t *lin,
	float64_t *c, float64_t *aicache, QP *qp, float64_t *epsilon_crit_target);

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
  void compute_matrices_for_optimization(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t *key,
	float64_t *a, float64_t *lin, float64_t *c, int32_t varnum, int32_t totdoc,
	float64_t *aicache, QP *qp);

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
  void compute_matrices_for_optimization_parallel(
	int32_t* docs, int32_t* label, int32_t *exclude_from_eq_const,
	float64_t eq_target, int32_t *chosen, int32_t *active2dnum, int32_t *key,
	float64_t *a, float64_t *lin, float64_t *c, int32_t varnum, int32_t totdoc,
	float64_t *aicache, QP *qp);

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
  int32_t   calculate_svm_model(
	int32_t* docs, int32_t *label,float64_t *lin, float64_t *a,
	float64_t* a_old, float64_t *c, int32_t *working2dnum, int32_t *active2dnum);

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
  int32_t   check_optimality(
	int32_t *label, float64_t *a, float64_t* lin, float64_t *c, int32_t totdoc,
	float64_t *maxdiff, float64_t epsilon_crit_org, int32_t *misclassified,
	int32_t *inconsistent,int32_t* active2dnum, int32_t *last_suboptimal_at,
	int32_t iteration);

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
  virtual void update_linear_component(
	int32_t* docs, int32_t *label, int32_t *active2dnum, float64_t *a,
	float64_t* a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
	float64_t *aicache, float64_t* c);

  /** helper for update linear component MKL linadd
   *
   * @param p p
   */
  static void* update_linear_component_mkl_linadd_helper(void* p);

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
  void update_linear_component_mkl(
		  int32_t* docs, int32_t *label, int32_t *active2dnum, float64_t *a,
		  float64_t* a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
		  float64_t *aicache);

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
  void update_linear_component_mkl_linadd(
		  int32_t* docs, int32_t *label, int32_t *active2dnum, float64_t *a,
		  float64_t* a_old, int32_t *working2dnum, int32_t totdoc, float64_t *lin,
		  float64_t *aicache);

  void call_mkl_callback(float64_t* a, int32_t* label, float64_t* lin);

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
  int32_t select_next_qp_subproblem_grad(
	int32_t *label, float64_t *a, float64_t* lin, float64_t* c, int32_t totdoc,
	int32_t qp_size, int32_t *inconsistent, int32_t* active2dnum,
	int32_t* working2dnum, float64_t *selcrit, int32_t *select,
	int32_t cache_only, int32_t *key, int32_t *chosen);

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
  int32_t select_next_qp_subproblem_rand(
	int32_t* label, float64_t *a, float64_t *lin, float64_t *c,
	int32_t totdoc, int32_t qp_size, int32_t *inconsistent,
	int32_t *active2dnum, int32_t *working2dnum, float64_t *selcrit,
	int32_t *select, int32_t *key, int32_t *chosen, int32_t iteration);

  /** select top n
   *
   * @param selcrit selcrit
   * @param range range
   * @param select select
   * @param n n
   */
  void   select_top_n(
	float64_t *selcrit, int32_t range, int32_t *select, int32_t n);

  /** init shrink state
   *
   * @param shrink_state shrink state
   * @param totdoc totdoc
   * @param maxhistory maximum history
   */
  void   init_shrink_state(
	SHRINK_STATE *shrink_state, int32_t totdoc, int32_t maxhistory);

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
  int32_t shrink_problem(
	SHRINK_STATE *shrink_state, int32_t *active2dnum,
	int32_t *last_suboptimal_at, int32_t iteration, int32_t totdoc,
	int32_t minshrink, float64_t *a, int32_t *inconsistent, float64_t* c,
	float64_t* lin, int* label);

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
  virtual void   reactivate_inactive_examples(
	int32_t *label,float64_t *a,SHRINK_STATE *shrink_state, float64_t *lin,
	float64_t *c, int32_t totdoc,int32_t iteration, int32_t *inconsistent,
	int32_t *docs,float64_t *aicache, float64_t* maxdiff);

protected:
	/** compute kernel
	*
	* @param i at index i
	* @param j at index j
	* @return computed kernel item at index i, j
	*/
	virtual float64_t compute_kernel(int32_t i, int32_t j)
	{
		return kernel->kernel(i, j);
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

	/** @return object name */
	virtual const char* get_name() const { return "SVMLight"; }

	/* interface to QP-solver */
	float64_t *optimize_qp( QP *qp,float64_t *epsilon_crit, int32_t nx,
			float64_t *threshold, int32_t& svm_maxqpsize);

	/** train SVM classifier
	 *
	 * @param data training data (parameter can be avoided if distance or
	 * kernel-based classifiers are used and distance/kernels are
	 * initialized with train data)
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

 protected:
  /** model */
  MODEL* model;
  /** learn parameters */
  LEARN_PARM* learn_parm;
  /** verbosity level (0-4) */
  int32_t   verbosity;

  /** init margin */
  float64_t init_margin;
  /** init iter */
  int32_t   init_iter;
  /** precision violations */
  int32_t precision_violations;
  /** model b */
  float64_t model_b;
  /** opt precision */
  float64_t opt_precision;
  /** primal */
  float64_t* primal;
  /** dual */
  float64_t* dual;

  // MKL stuff

  /** Matrix that stores the contribution by each kernel for each example (for
   * current alphas)
   */
  float64_t* W;
  /** number of iteration */
  int32_t count;
  /** current alpha gap */
  float64_t mymaxdiff;
  /** if kernel cache is used */
  bool use_kernel_cache;
  /** mkl converged */
  bool mkl_converged;
};
}
#endif //USE_SVMLIGHT
#endif //_SVMLight_H___
