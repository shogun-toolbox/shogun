/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CANBERRAWORDDISTANCE_H___
#define _CANBERRAWORDDISTANCE_H___

#include "lib/common.h"
#include "distance/StringDistance.h"

class CCanberraWordDistance: public CStringDistance<WORD>
{
 public:
  CCanberraWordDistance();
  ~CCanberraWordDistance();
  
  virtual bool init(CFeatures* l, CFeatures* r);
  virtual void cleanup();

  /// load and save kernel init_data
  bool load_init(FILE* src);
  bool save_init(FILE* dest);

  // return what type of distance we are CANBERRA,CHEBYSHEW, GEODESIC,...
  virtual EDistanceType get_distance_type() { return D_CANBERRAWORD; }

  // return the name of a distance
  virtual const CHAR* get_name() { return "CanberraWord"; }

  void get_dictionary(INT& dsize, DREAL*& dweights) 
  {
	  dsize=dictionary_size;
	  dweights = dictionary_weights;
  }

 protected:
  /// compute distance function for features a and b
  /// idx_{a,b} denote the index of the feature vectors
  /// in the corresponding feature object
  DREAL compute(INT idx_a, INT idx_b);

 protected:

  INT dictionary_size;
  DREAL* dictionary_weights;
};
#endif
