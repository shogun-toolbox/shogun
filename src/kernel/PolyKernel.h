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
  
  virtual void init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  virtual bool check_features(CFeatures* f);

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
  bool allow_init ;
  double scale ;
};

#endif

