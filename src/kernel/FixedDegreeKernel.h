#ifndef _FIXEDDEGREEKERNEL_H___
#define _FIXEDDEGREEKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CFixedDegreeKernel: public CKernel
{
 public:
  CFixedDegreeKernel(long size, int degree) ;
  ~CFixedDegreeKernel() ;
  
  virtual void init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  bool check_features(CFeatures* f) { return (f->get_feature_type()==F_STRING); }

  // return the name of a kernel
  virtual const char* get_name() { return "FixedDegree" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

 protected:
  int degree;
};

#endif
