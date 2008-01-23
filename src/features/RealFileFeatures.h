/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DREALFILEFEATURES__H__
#define _DREALFILEFEATURES__H__

#include "lib/common.h"
#include "features/RealFeatures.h"

class CRealFileFeatures: public CRealFeatures
{
 public:
  CRealFileFeatures(INT size, FILE* file);
  CRealFileFeatures(INT size, CHAR* filename);

  CRealFileFeatures(const CRealFileFeatures& orig);

  virtual ~CRealFileFeatures();
  
  virtual DREAL* load_feature_matrix();

  INT get_label(INT idx);

protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual DREAL* compute_feature_vector(INT num, INT& len, DREAL* target=NULL);

  bool load_base_data();

  FILE* working_file;
  CHAR* working_filename;
  bool status;
  INT* labels;

  BYTE intlen;
  BYTE doublelen;
  UINT endian;
  UINT fourcc;
  UINT preprocd;
  LONG filepos;
};
#endif
