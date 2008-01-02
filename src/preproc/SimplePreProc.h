/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSIMPLEPREPROC__H__
#define _CSIMPLEPREPROC__H__

#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

template <class ST> class CSimpleFeatures;

template <class ST> class CSimplePreProc : public CPreProc
{
	public:
		CSimplePreProc(const CHAR *name, const CHAR* id) : CPreProc(name,id) {}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual ST* apply_to_feature_matrix(CFeatures* f)=0;

		/// apply preproc on single feature vector
		/// result in feature matrix

		virtual ST* apply_to_feature_vector(ST* f, INT &len)=0;

		/// return that we are simple features (just fixed size matrices)
		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }
		/// return feature type
		inline virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CSimplePreProc<DREAL>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CSimplePreProc<SHORT>::get_feature_type()
{
	return F_SHORT;
}

template<> inline EFeatureType CSimplePreProc<WORD>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CSimplePreProc<CHAR>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CSimplePreProc<BYTE>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CSimplePreProc<ULONG>::get_feature_type()
{
	return F_ULONG;
}

#endif
