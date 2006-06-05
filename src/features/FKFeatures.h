#ifndef _CFKFEATURES__H__
#define _CFKFEATURES__H__

#include "features/RealFeatures.h"
#include "distributions/hmm/HMM.h"

class CFKFeatures: public CRealFeatures
{
 public:
  CFKFeatures(LONG size, CHMM* p, CHMM* n);
  CFKFeatures(const CFKFeatures &orig);
  
  virtual ~CFKFeatures();

  /// set HMMs and weight a
  void set_models(CHMM* p, CHMM* n);

  /// set weight a
  inline void set_a(DREAL a) 
  {
	  weight_a=a;
  }
  
  /// get weight a
  inline DREAL get_a() 
  {
	  return weight_a;
  }

  virtual DREAL* set_feature_matrix();
  
  virtual CFeatures* duplicate() const;

  double set_opt_a(double a=-1) ;
  inline double get_weight_a()  { return weight_a; };

 protected:
  virtual DREAL* compute_feature_vector(INT num, INT& len, DREAL* target=NULL);
  
  /// computes the featurevector to the address addr
  void compute_feature_vector(DREAL* addr, INT num, INT& len);
  
  double deriv_a(double a, INT dimension=-1) ;

 protected:
  CHMM* pos;
  CHMM* neg;
  double* pos_prob;
  double* neg_prob;
  DREAL weight_a;
};
#endif
