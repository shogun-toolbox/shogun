/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSTRINGPREPROC__H__
#define _CSTRINGPREPROC__H__

#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief Template class StringPreProc, base class for preprocessors (cf.
 * CPreProc) that apply to CStringFeatures (i.e. strings of variable length).
 *
 * Two new functions apply_to_string() and apply_to_string_features()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CStringFeatures.
 */
template <class ST> class CStringPreProc : public CPreProc
{
	public:
		/** constructor
		 *
		 * @param name string preprocessor's name
		 * @param id string preprocessor's id
		 */
		CStringPreProc(const char *name, const char* id) : CPreProc(name, id) {}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual bool apply_to_string_features(CFeatures* f)=0;

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, int32_t &len)=0;

		/// return that we are string features (just fixed size matrices)
		inline virtual EFeatureClass get_feature_class() { return C_STRING; }
		/// return feature type
		inline virtual EFeatureType get_feature_type();
};

template<> inline EFeatureType CStringPreProc<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> inline EFeatureType CStringPreProc<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> inline EFeatureType CStringPreProc<uint32_t>::get_feature_type()
{
	return F_UINT;
}

template<> inline EFeatureType CStringPreProc<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> inline EFeatureType CStringPreProc<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringPreProc<int16_t>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringPreProc<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringPreProc<int8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringPreProc<char>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CStringPreProc<bool>::get_feature_type()
{
	return F_BOOL;
}

template<> inline EFeatureType CStringPreProc<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> inline EFeatureType CStringPreProc<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CStringPreProc<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

}
#endif
