#ifndef _CBINSEQFEATURES__H__
#define _CBINSEQFEATURES__H__

#include "preproc/PreProc.h"
#include "features/ShortFeatures.h"
#include "hmm/Observation.h"

class CBinSeqFeatures: public CShortFeatures
{
 public:
  CBinSeqFeatures(CObservation *pos, CObservation *neg) ;
  virtual ~CBinSeqFeatures() ;

  virtual int get_label(long idx) ;
  virtual long get_number_of_examples() ;
  
protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual void compute_feature_vector(long num, short int* feat);

  CObservation *pos,*neg ;
};


#endif
