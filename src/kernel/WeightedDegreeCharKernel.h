#ifndef _WEIGHTEDDEGREECHARKERNEL_H___
#define _WEIGHTEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

class CWeightedDegreeCharKernel: public CCharKernel
{
 public:
  CWeightedDegreeCharKernel(long size, REAL* weights, int degree) ;
  ~CWeightedDegreeCharKernel() ;
  
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return the name of a kernel
  virtual const char* get_name() { return "FixedDegree" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

 protected:
  REAL* weights;
  int degree;
};

#endif
