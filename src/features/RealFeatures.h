#ifndef _DREALFEATURES__H__
#define _DREALFEATURES__H__

#include "features/SimpleFeatures.h"
#include "lib/common.h"
#include "features/CharFeatures.h"

class CRealFeatures: public CSimpleFeatures<DREAL>
{
 public:
  CRealFeatures(LONG size) : CSimpleFeatures<DREAL>(size)
  {
  }

  CRealFeatures(const CRealFeatures & orig) : CSimpleFeatures<DREAL>(orig)
  {
  }

  CRealFeatures(CHAR* fname) : CSimpleFeatures<DREAL>(fname)
  {
	load(fname);
  }

  bool Align_char_features(CCharFeatures* cf, CCharFeatures* Ref, DREAL gapCost) ;

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_DREAL; }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
 protected:
  DREAL Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, DREAL GapCost) ;

};
#endif
