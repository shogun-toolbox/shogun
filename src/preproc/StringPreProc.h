/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGPREPROC__H__
#define _CSTRINGPREPROC__H__

#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

#include <stdio.h>


template <class ST> class CStringFeatures;

template <class ST> class CStringPreProc : public CPreProc
{
public:
	CStringPreProc(const CHAR *name, const CHAR* id) : CPreProc(name,id)
	{
	}

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual bool apply_to_feature_strings(CFeatures* f)=0;

	/// apply preproc on single feature vector

	virtual ST* apply_to_feature_string(ST* f, INT &len)=0;

  /// return that we are simple minded features (just fixed size matrices)
  inline virtual EFeatureClass get_feature_class() { return C_STRING; }
  
};
#endif
