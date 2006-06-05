#ifndef _LINEARWORDKERNEL_H___
#define _LINEARWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"
#include "features/WordFeatures.h"

class CLinearWordKernel: public CWordKernel
{
 public:
  CLinearWordKernel(LONG size);
  ~CLinearWordKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_LINEAR; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Linear" ; } ;

  ///optimizable kernel, i.e. precompute normal vector and as phi(x)=x
  ///do scalar product in input space
  virtual bool init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas);
  virtual bool delete_optimization();
  virtual DREAL compute_optimized(INT idx);

  virtual void clear_normal();
  virtual void add_to_normal(INT idx, DREAL weight);

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  virtual void init_rescale();
  
 protected:
  double scale ;

  /// normal vector (used in case of optimized kernel)
  double* normal;
};

#endif
