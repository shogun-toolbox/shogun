#ifndef _REALFEATURES__H__
#define _REALFEATURES__H__

#include "features/SimpleFeatures.h"
#include "lib/common.h"
#include "features/CharFeatures.h"

class CRealFeatures: public CSimpleFeatures<REAL>
{
 public:
  CRealFeatures(LONG size) : CSimpleFeatures<REAL>(size)
  {
  }

  CRealFeatures(const CRealFeatures & orig) : CSimpleFeatures<REAL>(orig)
  {
  }

  CRealFeatures(CHAR* fname) : CSimpleFeatures<REAL>(fname)
  {
	load(fname);
  }

  bool Align_char_features(CCharFeatures* cf, CCharFeatures* Ref, REAL gapCost) ;

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_REAL; }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
 protected:
  REAL Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, REAL GapCost) ;

};
#endif
