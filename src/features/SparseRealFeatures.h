#ifndef _SPARSEREALFEATURES__H__
#define _SPARSEREALFEATURES__H__

#include "features/SparseFeatures.h"
#include "lib/common.h"

class CSparseRealFeatures: public CSparseFeatures<REAL>
{
 public:
  CSparseRealFeatures(LONG size) : CSparseFeatures<REAL>(size)
  {
  }

  CSparseRealFeatures(const CSparseRealFeatures & orig) : CSparseFeatures<REAL>(orig)
  {
  }

  CSparseRealFeatures(CHAR* fname) : CSparseFeatures<REAL>(fname)
  {
	load(fname);
  }

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_REAL; }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
};
#endif
