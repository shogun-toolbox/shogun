#ifndef _COMMWORDKERNEL_H___
#define _COMMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"

class CCommWordKernel: public CWordKernel
{
 public:
  CCommWordKernel(LONG size, bool use_sign) ;
  ~CCommWordKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_COMMWORD; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "CommWord" ; } ;

  virtual bool init_optimization(INT count, INT *IDX, REAL * weights) ;
  virtual bool delete_optimization() ;
  virtual REAL compute_optimized(INT idx) ;

  virtual void add_to_normal(INT idx, REAL weight);
  virtual void clear_normal();

  virtual void remove_lhs() ;
  virtual void remove_rhs() ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  REAL* sqrtdiag_lhs;
  REAL* sqrtdiag_rhs;

  bool initialized ;

  INT dictionary_size ;
  WORD * dictionary ;
  REAL * dictionary_weights ;
  
  bool use_sign ;
};

#endif
