%module WeightedDegreeCharKernel
%{
#include "kernel/WeightedDegreeCharKernel.h" 
%}

%include "lib/common.i"
%include "kernel/SimpleKernel.i"


%feature("notabstract") CWeightedDegreeCharKernel;


class CWeightedDegreeCharKernel: public CCharKernel
{
 public:
  CWeightedDegreeCharKernel(LONG size, DREAL* weights, INT degree, INT max_mismatch, bool use_normalization=true, bool block_computation=false, INT mkl_stepsize=1) ;
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

  virtual bool init_optimization(INT count, INT * IDX, DREAL * weights) ;
  virtual bool delete_optimization() ;
  virtual DREAL compute_optimized(INT idx) 
  { 
    if (get_is_initialized())
      return compute_by_tree(idx); 
    
    CIO::message(M_ERROR, "CWeightedDegreeCharKernel optimization not initialized\n") ;
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
  inline virtual void add_to_normal(INT idx, DREAL weight) 
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
  inline void compute_by_subkernel(INT idx, DREAL * subkernel_contrib)
	  { 
		  if (get_is_initialized())
		  {
			  compute_by_tree(idx, subkernel_contrib); 
			  return ;
		  }
		  CIO::message(M_ERROR, "CWeightedDegreePositionCharKernel optimization not initialized\n") ;
	  } ;
  inline const DREAL* get_subkernel_weights(INT& num_weights)
	  {
		  num_weights = get_num_subkernels() ;
		  
		  delete[] weights_buffer ;
		  weights_buffer = new DREAL[num_weights] ;
		  
		  if (position_weights!=NULL)
			  for (INT i=0; i<num_weights; i++)
				  weights_buffer[i] = position_weights[i*mkl_stepsize] ;
		  else
			  for (INT i=0; i<num_weights; i++)
				  weights_buffer[i] = weights[i*mkl_stepsize] ;
		  
		  return weights_buffer ;
	  }
  inline void set_subkernel_weights(DREAL* weights2, INT num_weights2)
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
  DREAL *compute_abs_weights(INT & len);
  void compute_by_tree(INT idx, DREAL *LevelContrib);

  bool is_tree_initialized() { return tree_initialized; }

  inline INT get_max_mismatch() { return max_mismatch; }
  inline INT get_degree() { return degree; }
  inline DREAL *get_degree_weights(INT& d, INT& len)
  {
	  d=degree;
	  len=length;
	  return weights;
  }
  inline DREAL *get_weights(INT& num_weights)
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
  inline DREAL *get_position_weights(INT& len)
  {
	  len=seq_length;
	  return position_weights;
  }

  bool set_weights(DREAL* weights, INT d, INT len=0);
  bool set_position_weights(DREAL* position_weights, INT len=0);
  bool init_matching_weights_wd();
  bool delete_position_weights() { delete[] position_weights ; position_weights=NULL ; return true ; } ;

 protected:

  void add_example_to_tree(INT idx, DREAL weight);
  void add_example_to_tree_mismatch(INT idx, DREAL weight);
  
  DREAL compute_by_tree(INT idx);

  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  DREAL compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) ;
  DREAL compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen);

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:

  ///degree*length weights
  ///length must match seq_length if != 0
  DREAL* weights;
  DREAL* position_weights ;
  DREAL* weights_buffer ;
  INT mkl_stepsize ;
  INT degree;
  INT length;
  
  INT max_mismatch ;
  INT seq_length ;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;

  bool tree_initialized ;
  bool use_normalization ;
  bool block_computation;

  DREAL* matching_weights;

#ifdef USE_TREEMEM
  INT TreeMemPtr ;
  INT TreeMemPtrMax ;
  
  inline void check_treemem()
	  {
		  if (TreeMemPtr+10>=TreeMemPtrMax) 
		  {
			  CIO::message(M_DEBUG, "Extending TreeMem from %i to %i elements\n", TreeMemPtrMax, (INT) ((double)TreeMemPtrMax*1.2)) ;
			  TreeMemPtrMax = (INT) ((double)TreeMemPtrMax*1.2) ;
		  } ;
	  } ;
#endif
};
