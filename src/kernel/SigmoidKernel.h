#ifndef _SIGMOIDKERNEL_H___
#define _SIGMOIDKERNEL_H___

#include "lib/common.h"
#include "kernel/RealKernel.h"
#include "features/RealFeatures.h"

class CSigmoidKernel: public CRealKernel
{
 public:
  CSigmoidKernel(LONG size, REAL gamma, REAL coef0);
  virtual ~CSigmoidKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_SIGMOID; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Sigmoid" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  double gamma;
  double coef0;
};
#endif
