#ifndef _POLYMATCHCHARKERNEL_H___
#define _POLYMATCHCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"
#include "features/CharFeatures.h"

class CPolyMatchCharKernel: public CCharKernel
{
 public:
  CPolyMatchCharKernel(LONG size, INT degree, bool inhomogene, bool use_normalization=true);
  ~CPolyMatchCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_POLYMATCH; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "PolyMatchChar"; };

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(INT idx_a, INT idx_b);

 protected:
  INT degree;
  bool inhomogene;

  double* sqrtdiag_lhs;
  double* sqrtdiag_rhs;

  bool initialized;
  bool use_normalization;
};

#endif
