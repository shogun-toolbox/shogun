#ifndef _LINEARBYTEKERNEL_H___
#define _LINEARBYTEKERNEL_H___

#include "lib/common.h"
#include "kernel/ByteKernel.h"
#include "features/ByteFeatures.h"

class CLinearByteKernel: public CByteKernel
{
 public:
  CLinearByteKernel(long size);
  ~CLinearByteKernel() ;
  
  virtual void init(CFeatures* l, CFeatures* r, bool do_init);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  virtual bool check_features(CFeatures* f);

  // return the name of a kernel
  virtual const char* get_name() { return "Linear" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual REAL compute(long idx_a, long idx_b);
  /*    compute_kernel*/

  virtual void init_rescale();
  
 protected:
  double scale ;
};

#endif

