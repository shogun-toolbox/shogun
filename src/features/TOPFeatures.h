#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/RealFeatures.h"
#include "hmm/HMM.h"

class CTOPFeatures: public CRealFeatures
{
 public:
  CTOPFeatures(CHMM* p, CHMM* n);
  CTOPFeatures(const CTOPFeatures &orig);
  
  virtual ~CTOPFeatures();

  void set_models(CHMM* p, CHMM* n);
  virtual REAL* set_feature_matrix();
  virtual int get_label(long idx);
  
  virtual CFeatures* duplicate() const;


 protected:
  virtual REAL* compute_feature_vector(long num, long& len, REAL* target=NULL);
  
  /// computes the featurevector to the address addr
  void compute_feature_vector(REAL* addr, long num, long& len);
  
 protected:
  CHMM* pos;
  CHMM* neg;
};
#endif
