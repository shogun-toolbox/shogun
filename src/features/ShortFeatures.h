#ifndef _SHORTFEATURES__H__
#define _SHORTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "lib/common.h"

class CShortFeatures: public CSimpleFeatures<SHORT>
{
 public:
  CShortFeatures(long size) : CSimpleFeatures<SHORT>(size)
  {
  }

  CShortFeatures(const CShortFeatures & orig) : CSimpleFeatures<SHORT>(orig)
  {
  }
  virtual EType get_feature_type() { return F_SHORT; }

  virtual bool load(char* fname);
  virtual bool save(char* fname);
};
#endif
