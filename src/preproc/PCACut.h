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

#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include <stdio.h>

#include "preproc/RealPreProc.h"
#include "features/Features.h"
#include "lib/common.h"


class CPCACut : public CRealPreProc
{
 public:
  CPCACut(INT do_whitening=0, double thresh=1e-6);
  virtual ~CPCACut();
  
  /// initialize preprocessor from features
  virtual bool init(CFeatures* f);
  /// initialize preprocessor from file
  virtual bool load_init_data(FILE* src);
  /// save init-data (like transforamtion matrices etc) to file
  virtual bool save_init_data(FILE* dst);
  /// cleanup
  virtual void cleanup();
  
  /// apply preproc on feature matrix
  /// result in feature matrix
  /// return pointer to feature_matrix, i.e. f->get_feature_matrix();
  virtual DREAL* apply_to_feature_matrix(CFeatures* f);
  
  /// apply preproc on single feature vector
  /// result in feature matrix
  virtual DREAL* apply_to_feature_vector(DREAL* f, INT &len);

 protected:
  double* T ;
  INT num_dim;
  INT num_old_dim;
  double *mean ;

  /// true when already initialized
  bool initialized;

  INT do_whitening;
  double thresh ;
};
#endif
#endif
