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

  CRealFeatures(char* fname) : CSimpleFeatures<REAL>(fname)
  {
	load(fname);
  }

  virtual CFeatures* duplicate() const;
  virtual EType get_feature_type() { return F_REAL; }

  virtual bool load(char* fname);
  virtual bool save(char* fname);
};
#endif
