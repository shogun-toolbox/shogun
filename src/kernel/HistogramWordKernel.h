#ifndef _HISTOGRAMWORDKERNEL_H___
#define _HISTOGRAMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/WordKernel.h"
#include "classifier/PluginEstimate.h"

class CHistogramWordKernel: public CWordKernel
{
 public:
  CHistogramWordKernel(LONG size, CPluginEstimate* pie);
  ~CHistogramWordKernel() ;
  
  virtual bool init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EKernelType get_kernel_type() { return K_HISTOGRAM; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "Histogram" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);
  //  DREAL compute_slow(LONG idx_a, LONG idx_b);

  inline INT compute_index(INT position, WORD symbol)
  {
	  return position*num_symbols+symbol+1;
  }

 protected:
  CPluginEstimate* estimate;

  DREAL* mean;
  DREAL* variance;

  DREAL* sqrtdiag_lhs;
  DREAL* sqrtdiag_rhs;

  DREAL* ld_mean_lhs;
  DREAL* ld_mean_rhs;

  DREAL* plo_lhs;
  DREAL* plo_rhs;

  INT num_params;
  INT num_params1;
  INT num_symbols;
  DREAL sum_m2_s2;

  bool initialized;
};

#endif
