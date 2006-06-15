/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
