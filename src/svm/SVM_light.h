#ifndef _SVMLight_H___
#define _SVMLight_H___

#include "svm/SVM.h"
#include "svm/kernel.h"
#include "svm/Optimizer.h"
#include "lib/Observation.h"
#include "lib/common.h"
#include "hmm/HMM.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h> 
#include <float.h>

# define VERSION       "V3.50"
# define VERSION_DATE  "01.11.00"


# define LINEAR  0           /* linear kernel type */
# define POLY    1           /* polynoial kernel type */
# define RBF     2           /* rbf kernel type */
# define SIGMOID 3           /* sigmoid kernel type */
# define TOP	 4           /* top kernel type */
# define LINEAR_TOP  5       /* linear top kernel type */

extern long   verbosity;              /* verbosity level (0-4) */
extern long   kernel_cache_statistic;
    

class CSVMLight:public CSVM
{
 public:
  
  CSVMLight();
  virtual ~CSVMLight();
  
  virtual bool svm_train(CObservation* train, int kernel_type, double C);
  virtual bool svm_test(CObservation* test, FILE* output);
  virtual bool load_svm(FILE* svm_file, CObservation* test);
  virtual bool save_svm(FILE* svm_file);
  
 private:
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
  
  typedef struct cache_parm_s {
    KERNEL_CACHE *kernel_cache;
    CFLOAT *cache;
    DOC *docs; 
    long m;
    KERNEL_PARM *kernel_parm;
    long offset,stepsize;
  } cache_parm_t;
  
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
    
 protected:
    double classify_example(MODEL *, DOC *);
    double classify_example_linear(MODEL *, DOC *);
    double model_length_s(MODEL *, KERNEL_PARM *);
    void   clear_vector_n(double *, long);
    //void   add_weight_vector_to_linear_model(MODEL *);
    void   read_model(char *, MODEL *, long, long);
    //void   read_documents(char *, DOC *, long *, long, long, long *, long *);
    //int    parse_document(char *, DOC *, long *, long *, long);
    //void   nol_ll(char *, long *, long *, long *);
    long   get_runtime();
    void   *my_malloc(long); 
    void   svm_learn(DOC *, long *, long, long, LEARN_PARM *, KERNEL_PARM *, 
	    KERNEL_CACHE *, MODEL *);
    long   optimize_to_convergence(DOC *, long *, long, long, LEARN_PARM *,
	    KERNEL_PARM *, KERNEL_CACHE *, SHRINK_STATE *,
	    MODEL *, long *, long *, double *, double *, 
	    TIMING *, double *, long, long);
    double compute_objective_function(double *, double *, long *, long *);
    void   clear_index(long *);
    void   add_to_index(long *, long);
    long   compute_index(long *,long, long *);
    void   optimize_svm(DOC *, long *, long *, long *, long *, MODEL *, long, 
	    long *, long, double *, double *, LEARN_PARM *, CFLOAT *, 
	    KERNEL_PARM *, QP *, double *);
    void   compute_matrices_for_optimization(DOC *, long *, long *, long *, 
	    long *, long *, MODEL *, double *, 
	    double *, long, long, LEARN_PARM *, 
	    CFLOAT *, KERNEL_PARM *, QP *);
    long   calculate_svm_model(DOC *, long *, long *, double *, double *, 
	    double *, LEARN_PARM *, long *, MODEL *);
    long   check_optimality(MODEL *, long *, long *, double *, double *,long, 
	    LEARN_PARM *,double *, double, long *, long *, long *,
	    long *, long, KERNEL_PARM *);
    long   identify_inconsistent(double *, long *, long *, long, LEARN_PARM *, 
	    long *, long *);
    long   identify_misclassified(double *, long *, long *, long,
	    MODEL *, long *, long *);
    long   identify_one_misclassified(double *, long *, long *, long,
	    MODEL *, long *, long *);
    long   incorporate_unlabeled_examples(MODEL *, long *,long *, long *,
	    double *, double *, long, double *,
	    long *, long *, long, KERNEL_PARM *,
	    LEARN_PARM *);
    void   update_linear_component(DOC *, long *, long *, double *, double *, 
	    long *, long, long, KERNEL_PARM *, 
	    KERNEL_CACHE *, double *,
	    CFLOAT *, double *);
    long   select_next_qp_subproblem_grad(long *, long *, double *, double *, long,
	    long, LEARN_PARM *, long *, long *, 
	    long *, double *, long *, KERNEL_CACHE *,
	    long *, long *);
    long   select_next_qp_subproblem_grad_cache(long *, long *, double *, double *,
	    long, long, LEARN_PARM *, long *, 
	    long *, long *, double *, long *,
	    KERNEL_CACHE *, long *, long *);
    void   select_top_n(double *, long, long *, long);
    void   init_shrink_state(SHRINK_STATE *, long, long);
    void   shrink_state_cleanup(SHRINK_STATE *);
    long   shrink_problem(LEARN_PARM *, SHRINK_STATE *, long *, long *, long,  
	    long, long, double *, long *);
    void   reactivate_inactive_examples(long *, long *, double *, SHRINK_STATE *,
	    double *, long, long, long, LEARN_PARM *, 
	    long *, DOC *, KERNEL_PARM *,
	    KERNEL_CACHE *, MODEL *, CFLOAT *, 
	    double *, double *);

    /* cache kernel evalutations to improve speed */
    void   get_kernel_row(KERNEL_CACHE *,DOC *, long, long, long *, CFLOAT *, 
	    KERNEL_PARM *);
    void   cache_kernel_row(KERNEL_CACHE *,DOC *, long, KERNEL_PARM *);
    void   cache_multiple_kernel_rows(KERNEL_CACHE *,DOC *, long *, long, 
	    KERNEL_PARM *);
    void   kernel_cache_shrink(KERNEL_CACHE *,long, long, long *);
    void   kernel_cache_init(KERNEL_CACHE *,long, long);
    void   kernel_cache_reset_lru(KERNEL_CACHE *);
    void   kernel_cache_cleanup(KERNEL_CACHE *);
    long   kernel_cache_malloc(KERNEL_CACHE *);
    void   kernel_cache_free(KERNEL_CACHE *,long);
    long   kernel_cache_free_lru(KERNEL_CACHE *);
    CFLOAT *kernel_cache_clean_and_malloc(KERNEL_CACHE *,long);
    long   kernel_cache_touch(KERNEL_CACHE *,long);
    long   kernel_cache_check(KERNEL_CACHE *,long);

    void compute_xa_estimates(MODEL *, long *, long *, long, DOC *, 
	    double *, double *, KERNEL_PARM *, 
	    LEARN_PARM *, double *, double *, double *);
    double xa_estimate_error(MODEL *, long *, long *, long, DOC *, 
	    double *, double *, KERNEL_PARM *, 
	    LEARN_PARM *);
    double xa_estimate_recall(MODEL *, long *, long *, long, DOC *, 
	    double *, double *, KERNEL_PARM *, 
	    LEARN_PARM *);
    double xa_estimate_precision(MODEL *, long *, long *, long, DOC *, 
	    double *, double *, KERNEL_PARM *, 
	    LEARN_PARM *);
    void avg_similarity_of_sv_of_one_class(MODEL *, DOC *, double *, long *, KERNEL_PARM *, double *, double *);
    double most_similar_sv_of_same_class(MODEL *, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *);
    double distribute_alpha_t_greedily(long *, long, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *, double);
    double distribute_alpha_t_greedily_noindex(MODEL *, DOC *, double *, long, long *, KERNEL_PARM *, LEARN_PARM *, double); 
    void estimate_transduction_quality(MODEL *, long *, long *, long, DOC *, double *);
    double estimate_margin_vcdim(MODEL *, double, double, KERNEL_PARM *);
    double estimate_sphere(MODEL *, KERNEL_PARM *);
    double estimate_r_delta_average(DOC *, long, KERNEL_PARM *); 
    double estimate_r_delta(DOC *, long, KERNEL_PARM *); 
    double length_of_longest_document_vector(DOC *, long, KERNEL_PARM *); 

    void   write_model(FILE *, MODEL *);
    void   write_prediction(char *, MODEL *, double *, double *, long *, long *,
	    long, LEARN_PARM *);
    void   write_alphas(char *, double *, long *, long);

protected:
    MODEL model ;
};

#endif
