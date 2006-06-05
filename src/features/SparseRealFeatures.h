#ifndef _SPARSEDREALFEATURES__H__
#define _SPARSEDREALFEATURES__H__

#include "features/SparseFeatures.h"
#include "lib/common.h"

class CSparseRealFeatures: public CSparseFeatures<DREAL>
{
 public:
  CSparseRealFeatures(LONG size) : CSparseFeatures<DREAL>(size)
  {
  }

  CSparseRealFeatures(const CSparseRealFeatures & orig) : CSparseFeatures<DREAL>(orig)
  {
  }

  CSparseRealFeatures(CHAR* fname) : CSparseFeatures<DREAL>(fname)
  {
	load(fname);
  }

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_DREAL; }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
};
#endif
