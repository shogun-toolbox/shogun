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
  CWeightedDegreeCharKernel(LONG size, REAL* weights, INT degree, INT max_mismatch, bool use_normalization=true) ;
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

  inline virtual void clear_normal()
  {
	  if (get_is_initialized())
	  {
		  delete_tree(NULL); 
		  set_is_initialized(false);
	  }
  }

  inline virtual void add_to_normal(INT idx, REAL weight) 
  {
	  add_example_to_tree(idx, weight);
	  set_is_initialized(true);
  }

  inline virtual INT get_num_subkernels()
  {
	  return get_degree();
  }

  // other kernel tree operations  
  void prune_tree(struct SuffixTree * p_tree=NULL, int min_usage=2);
  void count_tree_usage(INT idx);
  REAL *compute_abs_weights(INT & len);
  REAL compute_abs_weights_tree(struct SuffixTree * p_tree);
  void compute_by_tree(INT idx, REAL *LevelContrib);

  INT tree_size(struct SuffixTree * p_tree=NULL);
  bool is_tree_initialized() { return tree_initialized; }

  inline INT get_max_mismatch() { return max_mismatch; }
  inline INT get_degree() { return degree; }
  inline REAL *get_weights(INT& d, INT& len)
  {
	  d=degree;
	  len=length;
	  return weights;
  }

  void set_weights(REAL* weights, INT d, INT len=0);

 protected:

  void add_example_to_tree(INT idx, REAL weight);
  REAL compute_by_tree(INT idx);
  void delete_tree(struct SuffixTree * p_tree=NULL);

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:

  ///degree*length weights
  ///length must match seq_length if != 0
  REAL* weights;
  INT degree;
  INT length;

  INT max_mismatch ;
  INT seq_length ;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;
  bool *match_vector ;

  struct SuffixTree **trees ;
  bool tree_initialized ;
  bool use_normalization ;
  
};

#endif
