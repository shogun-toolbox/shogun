#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/RealKernel.h"
#include "features/RealFeatures.h"

class CLinearKernel: public CRealKernel
{
 public:
  CLinearKernel(long size, bool rescale) ;
  ~CLinearKernel() ;
  
  virtual void init(CRealFeatures* l, CRealFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  bool check_features(CFeatures* f);

  // return the name of a kernel
  virtual const char* get_name() { return "Linear" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

  void init_rescale();
  
 protected:
  bool rescale ;
  double scale ;
};

#endif

