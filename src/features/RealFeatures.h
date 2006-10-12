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
#include "features/CharFeatures.h"
#include "lib/common.h"

class CRealFeatures: public CSimpleFeatures<DREAL>
{
 public:
  CRealFeatures(INT size) : CSimpleFeatures<DREAL>(size)
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

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
 protected:
  DREAL Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, DREAL GapCost) ;

};
#endif
