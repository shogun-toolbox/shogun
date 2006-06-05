#ifndef _SIMPLELOCALITYIMPROVEDCHARKERNEL_H___
#define _SIMPLELOCALITYIMPROVEDCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/CharKernel.h"

class CSimpleLocalityImprovedCharKernel: public CCharKernel
{
 public:
  CSimpleLocalityImprovedCharKernel(LONG size, INT length, INT inner_degree, INT outer_degree) ;
  ~CSimpleLocalityImprovedCharKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_SIMPLELOCALITYIMPROVED; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "SimpleLocalityImproved" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

 protected:
  INT length;
  INT inner_degree;
  INT outer_degree;
  CHAR* match;
  REAL* pyramid_weights ;
};

#endif

