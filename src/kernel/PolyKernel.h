#ifndef _POLYKERNEL_H___
#define _POLYKERNEL_H___

#include "lib/common.h"
#include "kernel/RealKernel.h"
#include "features/RealFeatures.h"

class CPolyKernel: public CRealKernel
{
 public:
  CPolyKernel(long size, int degree, bool inhomogene);
  ~CPolyKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_POLY; }

  // return the name of a kernel
  virtual const char* get_name() { return "Poly" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

  virtual void init_rescale();
  
 protected:
  int degree;
  bool inhomogene ;
  double scale ;
};

#endif

