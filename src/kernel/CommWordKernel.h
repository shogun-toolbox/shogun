#ifndef _COMMWORDKERNEL_H___
#define _COMMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"

class CCommWordKernel: public CWordKernel
{
 public:
  CCommWordKernel(LONG size, bool use_sign, ENormalizationType normalization_=FULL_NORMALIZATION);
  ~CCommWordKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_COMMWORD; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "CommWord"; }

  virtual bool init_optimization(INT count, INT *IDX, DREAL * weights);
  virtual bool delete_optimization();
  virtual DREAL compute_optimized(INT idx);

  virtual void add_to_normal(INT idx, DREAL weight);
  virtual void clear_normal();

  virtual void remove_lhs();
  virtual void remove_rhs();

  void get_dictionary(INT& dsize, DREAL*& dweights) 
  {
	  dsize=dictionary_size;
	  dweights = dictionary_weights;
  }

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);

  inline DREAL normalize_weight(DREAL value, INT seq_num, INT seq_len, ENormalizationType normalization)
  {
	  switch (normalization)
	  {
		  case NO_NORMALIZATION:
			  return value;
			  break;
		  case SQRT_NORMALIZATION:
			  return value/sqrt(sqrtdiag_lhs[seq_num]);
			  break;
		  case FULL_NORMALIZATION:
			  return value/sqrtdiag_lhs[seq_num];
			  break;
		  case SQRTLEN_NORMALIZATION:
			  return value/sqrt(sqrt(seq_len));
			  break;
		  case LEN_NORMALIZATION:
			  return value/sqrt(seq_len);
			  break;
		  case SQLEN_NORMALIZATION:
			  return value/seq_len;
			  break;
		  default:
			  ASSERT(0);
	  }

	  return -CMath::INFTY;
  }

 protected:
  DREAL* sqrtdiag_lhs;
  DREAL* sqrtdiag_rhs;

  bool initialized;

  INT dictionary_size;
  DREAL* dictionary_weights;
  bool use_sign;

  ENormalizationType normalization;
};
#endif
