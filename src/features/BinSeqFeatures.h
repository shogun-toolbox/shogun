#ifndef _CBINSEQFEATURES__H__
#define _CBINSEQFEATURES__H__

#include "preproc/Preproc.h"
#include "features/ShortFeatures.h"
#include "hmm/Observation.h"

class CBinSeqFeatures: public CShortFeatures
{
 public:
  CBinSeqFeatures(CObservation *pos, CObservation *neg) ;
  ~CBinSeqFeatures() ;
  
protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual void compute_feature_vector(int num, short int* feat);

  CObservation *pos,*neg ;
};
#endif
