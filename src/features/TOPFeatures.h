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
      num_vectors=get_number_of_examples() ;
    };

  virtual REAL* set_feature_matrix();

  virtual int get_label(long idx) 
    {
	return pos->get_observations()->get_label(idx) ;
#warning check here
    } ;
  virtual long get_number_of_examples() 
    {
      if (pos)
	if (pos->get_observations())
	  return pos->get_observations()->get_DIMENSION();
	else 
	  return 0 ;
      else
	return 0;
#warning check here
    } ;

  virtual CFeatures* duplicate() const
    {
      return new CTOPFeatures(*this) ;
    }

 protected:
  virtual REAL* compute_feature_vector(long num, long& len);
  
  /// computes the featurevector to the address addr
  void compute_feature_vector(REAL* addr, long num, long& len);
  
 protected:
  CHMM* pos;
  CHMM* neg;
};
#endif
