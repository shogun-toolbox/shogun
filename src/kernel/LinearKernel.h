#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CLinearKernel: public CKernel
{
 public:
  CLinearKernel(long size, bool rescale) ;
  ~CLinearKernel() ;
  
  virtual void init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  bool check_features(CFeatures* f);

  // return the name of a kernel
  virtual const char* get_name() { return "Linear" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

  void init_rescale(CFeatures* f);
  
 protected:
  bool rescale ;
  double scale ;
};

#endif

