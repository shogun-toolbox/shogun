/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. */

#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CLinearKernel: public CKernel
{
 public:
  CLinearKernel(bool rescale) ;
  ~CLinearKernel() ;
  
  virtual void init(CFeatures* f);
  virtual void cleanup();
  
  virtual bool check_features(CFeatures* f) ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(CFeatures* a, int idx_a, CFeatures* b, int idx_b);
  /*    compute_kernel*/

  void init_rescale(CFeatures* f);
  
 protected:
  int rescale ;
  double scale ;
};

#endif

