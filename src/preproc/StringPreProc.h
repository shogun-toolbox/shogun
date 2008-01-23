/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGPREPROC__H__
#define _CSTRINGPREPROC__H__

#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

template <class ST> class CStringFeatures;

template <class ST> class CStringPreProc : public CPreProc
{
	public:
		CStringPreProc(const CHAR *name, const CHAR* id) : CPreProc(name,id) {}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual bool apply_to_string_features(CFeatures* f)=0;

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, INT &len)=0;

		/// return that we are string features (just fixed size matrices)
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }
		/// return feature type
		inline virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CStringPreProc<ULONG>::get_feature_type()
{
	return F_ULONG;
}

template<> inline EFeatureType CStringPreProc<WORD>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringPreProc<BYTE>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringPreProc<CHAR>::get_feature_type()
{
	return F_CHAR;
}

#endif
