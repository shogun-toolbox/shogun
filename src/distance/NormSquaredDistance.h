/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _NORMSQUAREDDISTANCE_H__
#define _NORMSQUAREDDISTANCE_H__

#include "lib/common.h"
#include "distance/RealDistance.h"
#include "features/RealFeatures.h"

class CNormSquaredDistance: public CRealDistance
{
 public:
  CNormSquaredDistance();
  CNormSquaredDistance(CRealFeatures* l, CRealFeatures* r);
  ~CNormSquaredDistance();
  
  virtual bool init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  virtual bool load_init(FILE* src);
  virtual bool save_init(FILE* dest);

  // return what type of kernel we are Linear,Polynomial, Gaussian,...
  virtual EDistanceType get_distance_type() { return D_NORMSQUARED; }

  /** return feature type the kernel can deal with
  */
  inline virtual EFeatureType get_feature_type() { return F_DREAL; }

  // return the name of a kernel
  virtual const CHAR* get_name() { return "NormSquared" ; } ;

 protected:
  /// compute kernel function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  virtual DREAL compute(INT idx_a, INT idx_b);
  /*    compute_kernel*/
  
 protected:
  double scale;
};

#endif
