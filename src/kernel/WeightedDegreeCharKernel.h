#ifndef _WEIGHTEDDEGREECHARKERNEL_H___
#define _WEIGHTEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

struct SuffixTree
{
	unsigned short has_floats ;
	unsigned short usage ;
	float weight ;
	union 
	{
		float child_weights[4] ;
		struct SuffixTree *childs[4] ;
	} ;
} ;

class CWeightedDegreeCharKernel: public CCharKernel
{
 public:
  CWeightedDegreeCharKernel(LONG size, REAL* weights, INT degree, INT max_mismatch) ;
  ~CWeightedDegreeCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREE; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "WeightedDegree" ; } ;

  virtual bool init_optimization(INT count, INT * IDX, REAL * weights) ;
  virtual bool delete_optimization() ;
  virtual REAL compute_optimized(INT idx) 
	  { 
		  if (get_is_initialized())
			  return compute_by_tree(idx); 

		  CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not initialized\n") ;
		  return 0 ;
	  } ;

  // other kernel tree operations  
  void prune_tree(struct SuffixTree * p_tree=NULL, int min_usage=2) ;
  void count_tree_usage(INT idx)  ;
  REAL *compute_abs_weights(INT & len)  ;
  REAL compute_abs_weights_tree(struct SuffixTree * p_tree)  ;

  INT tree_size(struct SuffixTree * p_tree=NULL) ;
  bool is_tree_initialized() { return tree_initialized ; } ;
  INT get_max_mismatch() { return max_mismatch ; } ;

 protected:

  void add_example_to_tree(INT idx, REAL weight) ;
  REAL compute_by_tree(INT idx) ;
  void delete_tree(struct SuffixTree * p_tree=NULL) ;

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:
  REAL* weights;
  INT degree;
  INT max_mismatch ;
  INT seq_length ;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;
  bool *match_vector ;

  struct SuffixTree **trees ;
  bool tree_initialized ;
  
};

#endif
