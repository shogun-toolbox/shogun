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

  CRealFeatures(DREAL* feature_matrix, INT num_feat, INT num_vec) : CSimpleFeatures<DREAL>(feature_matrix, num_feat, num_vec)
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
