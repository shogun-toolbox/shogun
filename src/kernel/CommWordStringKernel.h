#ifndef _COMMWORDSTRINGKERNEL_H___
#define _COMMWORDSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

enum E_NormalizationType
{
	E_NO_NORMALIZATION,
	E_SQRT_NORMALIZATION,
	E_FULL_NORMALIZATION,
	E_SQRTLEN_NORMALIZATION,
	E_LEN_NORMALIZATION,
	E_SQLEN_NORMALIZATION 
} ;

class CCommWordStringKernel: public CStringKernel<WORD>
{
 public:
  CCommWordStringKernel(LONG size, bool use_sign, E_NormalizationType normalization_=E_FULL_NORMALIZATION ) ;
  ~CCommWordStringKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_COMMWORDSTRING; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "CommWordString" ; } ;

  virtual bool init_optimization(INT count, INT *IDX, REAL * weights) ;
  virtual bool delete_optimization() ;
  virtual REAL compute_optimized(INT idx) ;

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

  inline virtual EFeatureType get_feature_type() { return F_WORD; }
  
  void get_dictionary(INT &dsize, WORD*& dict, REAL*& dweights) 
	  {
		  dsize=dictionary_size ;
		  dict=dictionary ;
		  dweights = dictionary_weights ;
	  } ;
  
 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized ;

  INT dictionary_size ;
  WORD * dictionary ;
  REAL * dictionary_weights ;
  
  bool use_sign ;
  E_NormalizationType normalization ;
  
};

#endif
