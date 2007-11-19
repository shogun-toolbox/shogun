/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Fabio De Bona
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DREALFEATURES__H__
#define _DREALFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CRealFeatures: public CSimpleFeatures<DREAL>
{
 public:
  CRealFeatures(INT size=0) : CSimpleFeatures<DREAL>(size)
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

  inline virtual void get_fm(DREAL** dst, INT* d1, INT* d2)
  {
      CSimpleFeatures<DREAL>::get_fm(dst, d1, d2);
  }
  inline virtual void copy_feature_matrix(DREAL* src, INT num_feat, INT num_vec)
  {
      CSimpleFeatures<DREAL>::copy_feature_matrix(src, num_feat, num_vec);
  }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
};
#endif
