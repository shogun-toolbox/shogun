#ifndef _REALFEATURES__H__
#define _REALFEATURES__H__

#include "features/SimpleFeatures.h"
#include "lib/common.h"

class CRealFeatures: public CSimpleFeatures<REAL>
{
 public:
  CRealFeatures(long size) : CSimpleFeatures<REAL>(size)
  {
  }

  CRealFeatures(const CRealFeatures & orig) : CSimpleFeatures<REAL>(orig)
  {
  }
  virtual EType get_feature_type() { return F_REAL; }

  virtual bool load(FILE* dest);
  virtual bool save(FILE* dest);
};
#endif
