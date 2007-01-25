/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MINKOWSKI_H___
#define _MINKOWSKI_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

class CMinkowskiMetric: public CSimpleDistance<DREAL>
{
 public:
  CMinkowskiMetric(DREAL k);
  virtual ~CMinkowskiMetric();
  
  virtual bool init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return type of distance
  virtual EDistanceType get_distance_type() { return D_MINKOWSKI;}

  // return the name of distance
  virtual const CHAR* get_name() { return "Minkowski-Metric"; };

 protected:
  /// compute distance for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual DREAL compute(INT idx_a, INT idx_b);

 protected:
  DREAL k;
};

#endif
