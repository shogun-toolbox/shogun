#ifndef _LINEARKERNEL_H___
#define _LINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/RealKernel.h"
#include "features/RealFeatures.h"

class CLinearKernel: public CRealKernel
{
 public:
  CLinearKernel(LONG size);
  ~CLinearKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_LINEAR; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Linear" ; } ;

  virtual bool init_optimization(INT count, INT *IDX, REAL * weights) 
	  {
		  CIO::message(M_ERROR, "not implemented yet") ;
		  return false;
	  } ;
  virtual void delete_optimization() {} ;
  virtual REAL compute_optimized(INT idx) 
	  { 		  
		  CIO::message(M_ERROR, "not implemented yet") ;
		  return 0 ; 
	  } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/

  virtual void init_rescale();
  
 protected:
  double scale ;
};

#endif
