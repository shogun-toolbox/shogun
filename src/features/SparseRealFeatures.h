#ifndef _SPARSEREALFEATURES__H__
#define _SPARSEREALFEATURES__H__

#include "features/SparseFeatures.h"
#include "lib/common.h"

class CSparseRealFeatures: public CSparseFeatures<REAL>
{
 public:
  CSparseRealFeatures(long size) : CSparseFeatures<REAL>(size)
  {
  }

  CSparseRealFeatures(const CSparseRealFeatures & orig) : CSparseFeatures<REAL>(orig)
  {
  }

  CSparseRealFeatures(char* fname) : CSparseFeatures<REAL>(fname)
  {
	load(fname);
  }

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_REAL; }

  virtual bool load(char* fname);
  virtual bool save(char* fname);
};
#endif
