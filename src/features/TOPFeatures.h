#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include "features/RealFeatures.h"
#include "hmm/HMM.h"

class CTOPFeatures: public CRealFeatures
{
 public:
  CTOPFeatures(CHMM* p, CHMM* n);
  CTOPFeatures(const CTOPFeatures &orig): CRealFeatures(orig), pos(orig.pos), neg(orig.neg)
    { } ;

  virtual ~CTOPFeatures();

  void set_models(CHMM* p, CHMM* n)
    {
      pos=p ; 
      neg=n ;
      delete[] feature_matrix  ;
      feature_matrix=NULL ;
      //set_feature_matrix() ;
    };

  virtual REAL* set_feature_matrix();

  virtual int get_label(long idx) 
    {
	return pos->get_observations()->get_label(idx) ;
#warning check here
    } ;
  virtual long get_number_of_examples() 
    {
	return pos->get_observations()->get_DIMENSION();
#warning check here
    } ;

  virtual CFeatures* duplicate() const
    {
      return new CTOPFeatures(*this) ;
    }

 protected:
  virtual REAL* compute_feature_vector(int num, int& len);
  
  /// computes the featurevector to the address addr
  void compute_feature_vector(REAL* addr, int num, int& len);
  
 protected:
  CHMM* pos;
  CHMM* neg;
};
#endif
