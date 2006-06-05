#ifndef _FIXEDDEGREECHARKERNEL_H___
#define _FIXEDDEGREECHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

class CFixedDegreeCharKernel: public CCharKernel
{
 public:
  CFixedDegreeCharKernel(LONG size, INT degree) ;
  ~CFixedDegreeCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_FIXEDDEGREE; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "FixedDegree" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  INT degree;

  REAL* sqrtdiag_lhs;
  REAL* sqrtdiag_rhs;

  bool initialized ;
};

#endif
