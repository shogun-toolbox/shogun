#ifndef _SPARSEGAUSSIANKERNEL_H___
#define _SPARSEGAUSSIANKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseRealKernel.h"
#include "features/SparseRealFeatures.h"

class CSparseGaussianKernel: public CSparseRealKernel
{
 public:
  CSparseGaussianKernel(LONG size, double width);
  ~CSparseGaussianKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_GAUSSIAN; }

  /** return feature type the kernel can deal with
  */
  inline virtual EFeatureType get_feature_type() { return F_REAL; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "SparseGaussian" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  
 protected:
  double width;
};

#endif

