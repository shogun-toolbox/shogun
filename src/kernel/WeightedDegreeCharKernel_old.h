#ifndef _WEIGHTEDDEGREECHARKERNELOLD_H___
#define _WEIGHTEDDEGREECHARKERNELOLD_H___

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

class CWeightedDegreeCharKernel_old: public CCharKernel
{
 public:
  CWeightedDegreeCharKernel_old(LONG size, REAL* weights, INT degree, INT max_mismatch, bool use_normalization=true) ;
  ~CWeightedDegreeCharKernel_old() ;
  
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

		  CIO::message(M_ERROR, "CWeightedDegreeCharKernel_old optimization not initialized\n") ;
		  return 0 ;
	  } ;

  // subkernel functionality
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
	  if (position_weights!=NULL)
		  return seq_length ;
	  if (length==0)
		  return get_degree();
	  return get_degree()*length ;
  }
  inline void compute_by_subkernel(INT idx, REAL * subkernel_contrib)
	  { 
		  if (get_is_initialized())
		  {
			  compute_by_tree(idx, subkernel_contrib); 
			  return ;
		  }
		  CIO::message(M_ERROR, "CWeightedDegreePositionCharKernel optimization not initialized\n") ;
	  } ;
  inline const REAL* get_subkernel_weights(INT& num_weights)
	  {
		  return get_weights(num_weights) ;
	  }
  inline void set_subkernel_weights(REAL* weights2, INT num_weights2)
	  {
		  INT num_weights=-1 ;
		  REAL* weights = get_weights(num_weights) ;
		  if (num_weights!=num_weights2)
			  CIO::message(M_ERROR, "number of weights do not match\n") ;
		  for (INT i=0; i<num_weights; i++)
			  weights[i]=weights2[i] ;
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
  inline REAL *get_degree_weights(INT& d, INT& len)
  {
	  d=degree;
	  len=length;
	  return weights;
  }
  inline REAL *get_weights(INT& num_weights)
  {
	  if (position_weights!=NULL)
	  {
		  num_weights = seq_length ;
		  return position_weights ;
	  }
	  if (length==0)
		  num_weights = degree ;
	  else
		  num_weights = degree*length ;
	  return weights;
  }
  inline REAL *get_position_weights(INT& len)
  {
	  len=seq_length;
	  return position_weights;
  }

  bool set_weights(REAL* weights, INT d, INT len=0);
  bool set_position_weights(REAL* position_weights, INT len=0);
  bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;

 protected:

  void add_example_to_tree(INT idx, REAL weight);
  REAL compute_by_tree(INT idx);
  void delete_tree(struct SuffixTree * p_tree=NULL);

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  REAL compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  REAL compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  REAL compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:

  ///degree*length weights
  ///length must match seq_length if != 0
  REAL* weights;
  REAL* position_weights ;
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
