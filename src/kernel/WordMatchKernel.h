#ifndef _WORDMATCHKERNEL_H___
#define _WORDMATCHKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"
#include "features/WordFeatures.h"

class CWordMatchKernel: public CWordKernel
{
 public:
  CWordMatchKernel(LONG size, INT degree);
  ~CWordMatchKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_LINEAR; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Linear" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  virtual void init_rescale();
  
 protected:
  double scale ;
  INT degree;
};

#endif
