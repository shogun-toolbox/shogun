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

#include <shogun/lib/config.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/Preprocessor.h>

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief Template class StringPreprocessor, base class for preprocessors (cf.
 * CPreprocessor) that apply to CStringFeatures (i.e. strings of variable length).
 *
 * Two new functions apply_to_string() and apply_to_string_features()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CStringFeatures.
 */
template <class ST> class CStringPreprocessor : public CPreprocessor
{
	public:
		/** constructor
		 */
		CStringPreprocessor() : CPreprocessor() {}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual bool apply_to_string_features(CFeatures* f)=0;

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, int32_t &len)=0;

		/// return that we are string features (just fixed size matrices)
		virtual EFeatureClass get_feature_class() { return C_STRING; }
		/// return feature type
		virtual EFeatureType get_feature_type();

		/// return the name of the preprocessor
		virtual const char* get_name() const { return "UNKNOWN"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_UNKNOWN; }

};

template<> inline EFeatureType CStringPreprocessor<uint64_t>::get_feature_type()
{
	return F_ULONG;
}

template<> inline EFeatureType CStringPreprocessor<int64_t>::get_feature_type()
{
	return F_LONG;
}

template<> inline EFeatureType CStringPreprocessor<uint32_t>::get_feature_type()
{
	return F_UINT;
}

template<> inline EFeatureType CStringPreprocessor<int32_t>::get_feature_type()
{
	return F_INT;
}

template<> inline EFeatureType CStringPreprocessor<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringPreprocessor<int16_t>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CStringPreprocessor<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringPreprocessor<int8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CStringPreprocessor<char>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CStringPreprocessor<bool>::get_feature_type()
{
	return F_BOOL;
}

template<> inline EFeatureType CStringPreprocessor<float32_t>::get_feature_type()
{
	return F_SHORTREAL;
}

template<> inline EFeatureType CStringPreprocessor<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CStringPreprocessor<floatmax_t>::get_feature_type()
{
	return F_LONGREAL;
}

}
#endif
