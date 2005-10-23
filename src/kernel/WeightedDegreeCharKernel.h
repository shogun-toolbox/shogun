#ifndef _WEIGHTEDDEGREECHARKERNEL_H___
#define _WEIGHTEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

//#define USE_TREEMEM

#ifdef USE_TREEMEM
#define NO_CHILD ((INT)-1) 
#else
#define NO_CHILD NULL
#endif

struct Trie
{
  unsigned short has_floats ;
  unsigned short usage ;
  float weight ;
  union 
  {
    float child_weights[4] ;
#ifdef USE_TREEMEM
    INT childs[4] ; // int32 should be sufficient
#else
    struct Trie *childs[4] ;
#endif
  } ;
} ;

class CWeightedDegreeCharKernel: public CCharKernel
{
 public:
  CWeightedDegreeCharKernel(LONG size, REAL* weights, INT degree, INT max_mismatch, bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1) ;
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
  virtual INT compute_optimized_active(LONG start, LONG end, LONG *active_idx, LONG *example_idx, REAL *active_output) ;
  REAL compute_optimized_active_helper(INT idx, INT tree_idx) ;

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
	  if (max_mismatch==0)
		  add_example_to_tree(idx, weight);
	  else
		  add_example_to_tree_mismatch(idx, weight);

	  set_is_initialized(true);
  }
  inline virtual INT get_num_subkernels()
	  {
		  //fprintf(stderr, "mkl_stepsize=%i\n", mkl_stepsize) ;
		  //exit(-1) ;
		  if (position_weights!=NULL)
			  return (INT) ceil(1.0*seq_length/mkl_stepsize) ;
		  if (length==0)
			  return (INT) ceil(1.0*get_degree()/mkl_stepsize);
		  return (INT) ceil(1.0*get_degree()*length/mkl_stepsize) ;
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
		  num_weights = get_num_subkernels() ;
		  
		  delete[] weights_buffer ;
		  weights_buffer = new REAL[num_weights] ;
		  
		  if (position_weights!=NULL)
			  for (INT i=0; i<num_weights; i++)
				  weights_buffer[i] = position_weights[i*mkl_stepsize] ;
		  else
			  for (INT i=0; i<num_weights; i++)
				  weights_buffer[i] = weights[i*mkl_stepsize] ;
		  
		  return weights_buffer ;
	  }
  inline void set_subkernel_weights(REAL* weights2, INT num_weights2)
	  {
		  INT num_weights = get_num_subkernels() ;
		  if (num_weights!=num_weights2)
			  CIO::message(M_ERROR, "number of weights do not match\n") ;
		  
		  if (position_weights!=NULL)
			  for (INT i=0; i<num_weights; i++)
				  for (INT j=0; j<mkl_stepsize; j++)
				  {
					  if (i*mkl_stepsize+j<seq_length)
						  position_weights[i*mkl_stepsize+j] = weights2[i] ;
				  }
		  else if (length==0)
		  {
			  for (INT i=0; i<num_weights; i++)
				  for (INT j=0; j<mkl_stepsize; j++)
					  if (i*mkl_stepsize+j<get_degree())
						  weights[i*mkl_stepsize+j] = weights2[i] ;
		  }
		  else
		  {
			  for (INT i=0; i<num_weights; i++)
				  for (INT j=0; j<mkl_stepsize; j++)
					  if (i*mkl_stepsize+j<get_degree()*length)
						  weights[i*mkl_stepsize+j] = weights2[i] ;
		  }
	  }
  
  // other kernel tree operations  
  void prune_tree(struct Trie * p_tree=NULL, int min_usage=2);
  void count_tree_usage(INT idx);
  REAL *compute_abs_weights(INT & len);
  REAL compute_abs_weights_tree(struct Trie * p_tree);
  void compute_by_tree(INT idx, REAL *LevelContrib);

  INT tree_size(struct Trie * p_tree=NULL);
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
  bool init_matching_weights_wd();
  bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;

 protected:

  void add_example_to_tree(INT idx, REAL weight);
  void add_example_to_tree_mismatch(INT idx, REAL weight);
  void add_example_to_tree_mismatch_recursion(struct Trie *tree,  REAL alpha,
											  INT *vec, INT len_rem, 
											  INT depth_rec, INT mismatch_rec) ;
  
  REAL compute_by_tree(INT idx);
  void delete_tree(struct Trie * p_tree=NULL);

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  REAL compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  REAL compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  REAL compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  REAL compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen);

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:

  ///degree*length weights
  ///length must match seq_length if != 0
  REAL* weights;
  REAL* position_weights ;
  REAL* weights_buffer ;
  INT mkl_stepsize ;
  INT degree;
  INT length;
  
  INT max_mismatch ;
  INT seq_length ;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;

  struct Trie **trees ;
  bool tree_initialized ;
  bool use_normalization ;
  bool block_computation;

  REAL* matching_weights;

#ifdef USE_TREEMEM
  struct Trie* TreeMem ;
  INT TreeMemPtr ;
  INT TreeMemPtrMax ;
  
  inline void check_treemem()
    {
      if (TreeMemPtr+10>=TreeMemPtrMax) 
	{
	  TreeMemPtrMax = (INT) ((double)TreeMemPtr*1.2) ;
	  TreeMem = (struct Trie *)realloc(TreeMem,TreeMemPtrMax*sizeof(struct Trie)) ;
	} ;
    } ;

#endif
};
#endif
