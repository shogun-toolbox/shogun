#ifndef _COMMWORDKERNEL_H___
#define _COMMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"

class CCommWordKernel: public CWordKernel
{
 public:
  CCommWordKernel(LONG size) ;
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
};

#endif
