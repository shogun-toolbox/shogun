#ifndef _HistogramKernel_H___
#define _HistogramKernel_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CHistogramKernel: public CKernel
{
 public:
  CHistogramKernel(long size, int degree) ;
  ~CHistogramKernel() ;
  
  virtual void init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  bool check_features(CFeatures* f) { return (f->get_feature_type()==F_STRING); }

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
  int degree;
};

#endif

