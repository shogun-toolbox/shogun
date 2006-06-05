#ifndef _MANHATTENWORDKERNEL_H___
#define _MANHATTENWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"

class CManhattenWordKernel: public CWordKernel
{
 public:
  CManhattenWordKernel(LONG size, DREAL width);
  ~CManhattenWordKernel();
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_MANHATTENWORD; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "ManhattenWord"; }

  void get_dictionary(INT& dsize, DREAL*& dweights) 
  {
	  dsize=dictionary_size;
	  dweights = dictionary_weights;
  }

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);

 protected:

  INT dictionary_size;
  DREAL* dictionary_weights;
  DREAL width;
};
#endif
