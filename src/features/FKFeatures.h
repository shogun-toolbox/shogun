#ifndef _CFKFEATURES__H__
#define _CFKFEATURES__H__

#include "features/RealFeatures.h"
#include "hmm/HMM.h"

class CFKFeatures: public CRealFeatures
{
 public:
  CFKFeatures(long size, CHMM* p, CHMM* n, REAL a);
  CFKFeatures(const CFKFeatures &orig);
  
  virtual ~CFKFeatures();

  /// set HMMs and weight a
  void set_models(CHMM* p, CHMM* n, REAL a);

  /// set weight a
  inline void set_a(REAL a) 
  {
	  weight_a=a;
  }
  
  /// get weight a
  inline REAL get_a() 
  {
	  return weight_a;
  }

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
  REAL weight_a;
};
#endif
