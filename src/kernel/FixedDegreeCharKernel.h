#ifndef _FIXEDDEGREECHARKERNEL_H___
#define _FIXEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

class CFixedDegreeCharKernel: public CCharKernel
{
 public:
  CFixedDegreeCharKernel(long size, int degree) ;
  ~CFixedDegreeCharKernel() ;
  
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
  int degree;
};

#endif
