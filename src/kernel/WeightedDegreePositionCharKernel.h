#ifndef _WEIGHTEDDEGREEPOSITIONCHARKERNEL_H___
#define _WEIGHTEDDEGREEPOSITIONCHARKERNEL_H___


#include "lib/common.h"
#include "kernel/CharKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"

class CWeightedDegreePositionCharKernel: public CCharKernel
{
 public:
  CWeightedDegreePositionCharKernel(LONG size, REAL* weights, INT degree, INT max_mismatch, INT * shift, INT shift_len) ;
  ~CWeightedDegreePositionCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPOS; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "WeightedDegreePos" ; } ;

  virtual bool init_optimization(INT count, INT *IDX, REAL * weights) ;
  virtual void delete_optimization() ;
  virtual REAL compute_optimized(INT idx) 
	  { 
		  if (get_is_initialized())
			  return compute_by_tree(idx); 
		  
		  CIO::message(M_ERROR, "CWeightedDegreePositionCharKernel optimization not initialized\n") ;
		  return 0 ;
	  } ;
  
  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  REAL compute2(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  void add_example_to_tree(INT idx, REAL weight) ;
  void delete_tree(struct SuffixTree * p_tree=NULL) ;

  REAL compute_by_tree(INT idx) ;
  /* computes the simple kernel between position seq_pos and tree tree_pos */
  REAL compute_by_tree_helper(INT* vec, INT seq_pos, INT tree_pos)
	  {
		  REAL sum=0 ;
		  struct SuffixTree *tree = trees[tree_pos] ;
		  assert(tree!=NULL) ;
		  
		  for (INT j=0; j<degree; j++)
		  {
			  if ((!tree->has_floats) && (tree->childs[vec[seq_pos+j]]!=NULL))
			  {
				  tree=tree->childs[vec[seq_pos+j]] ;
				  sum += tree->weight ;
			  } else
				  if (tree->has_floats)
				  {
					  sum += tree->child_weights[vec[seq_pos+j]] ;
					  break ;
				  } else
					  break ;
		  } 
		  return sum ;
	  } ;
  
 protected:
  REAL* weights;
  INT * counts ;
  INT degree;
  INT max_mismatch ;
  INT *shift ;
  INT shift_len ;
  INT max_shift ;

  INT seq_length ;
  
  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;

  bool *match_vector ;
  struct SuffixTree **trees ;
  bool tree_initialized ;
};

#endif
