#ifndef _COMMWORDSTRINGKERNEL_H___
#define _COMMWORDSTRINGKERNEL_H___

#include "lib/common.h"
#include "lib/Mathmatics.h"
#include "kernel/StringKernel.h"

class CCommWordStringKernel: public CStringKernel<WORD>
{
 public:
  CCommWordStringKernel(LONG size, bool use_sign, E_NormalizationType normalization_=E_FULL_NORMALIZATION );
  ~CCommWordStringKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_COMMWORDSTRING; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "CommWordString"; }

  virtual bool init_optimization(INT count, INT *IDX, REAL * weights);
  virtual bool delete_optimization();
  virtual REAL compute_optimized(INT idx);

  virtual void add_to_normal(INT idx, REAL weight);
  virtual void clear_normal();

  virtual void remove_lhs();
  virtual void remove_rhs();

  inline virtual EFeatureType get_feature_type() { return F_WORD; }
  
  void get_dictionary(INT& dsize, REAL*& dweights) 
  {
	  dsize=dictionary_size;
	  dweights = dictionary_weights;
  }

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);

  inline REAL normalize_weight(REAL value, INT seq_num, INT seq_len, E_NormalizationType normalization)
  {
	  switch (normalization)
	  {
		  case E_NO_NORMALIZATION:
			  return value;
			  break;
		  case E_SQRT_NORMALIZATION:
			  return value/sqrt(sqrtdiag_lhs[seq_num]);
			  break;
		  case E_FULL_NORMALIZATION:
			  return value/sqrtdiag_lhs[seq_num];
			  break;
		  case E_SQRTLEN_NORMALIZATION:
			  return value/sqrt(sqrt(seq_len));
			  break;
		  case E_LEN_NORMALIZATION:
			  return value/sqrt(seq_len);
			  break;
		  case E_SQLEN_NORMALIZATION:
			  return value/seq_len;
			  break;
		  default:
			  assert(0);
	  }

	  return -CMath::INFTY;
  }

 protected:
  REAL* sqrtdiag_lhs;
  REAL* sqrtdiag_rhs;

  bool initialized;

  INT dictionary_size;
  REAL* dictionary_weights;
  bool use_sign;

  E_NormalizationType normalization;
};
#endif
