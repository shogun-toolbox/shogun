/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CSIMPLEPREPROC__H__
#define _CSIMPLEPREPROC__H__

#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/common.h"
#include "preproc/PreProc.h"

namespace shogun
{
template <class ST> class CSimpleFeatures;

/** @brief Template class SimplePreProc, base class for preprocessors (cf.
 * CPreProc) that apply to CSimpleFeatures (i.e. rectangular dense matrices)
 *
 * Two new functions apply_to_feature_vector() and apply_to_feature_matrix()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CSimpleFeatures. For examples see e.g. CLogPlusOne
 * or CPCACut.
 */
template <class ST> class CSimplePreProc : public CPreProc
{
	public:
		/** constructor
		 *
		 * @param name simple preprocessor's name
		 * @param id simple preprocessor's id
		 */
		CSimplePreProc() : CPreProc() {}

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual ST* apply_to_feature_matrix(CFeatures* f)=0;

		/// apply preproc on single feature vector
		/// result in feature matrix

		virtual ST* apply_to_feature_vector(ST* f, int32_t &len)=0;

		/// return that we are simple features (just fixed size matrices)
		virtual inline EFeatureClass get_feature_class() { return C_SIMPLE; }
		/// return feature type
		virtual inline EFeatureType get_feature_type();

		/// return a type of preprocessor
		virtual inline EPreProcType get_type() const { return P_UNKNOWN; }

};

template<> inline EFeatureType CSimplePreProc<float64_t>::get_feature_type()
{
	return F_DREAL;
}

template<> inline EFeatureType CSimplePreProc<int16_t>::get_feature_type()
{
	return F_SHORT;
}

template<> inline EFeatureType CSimplePreProc<uint16_t>::get_feature_type()
{
	return F_WORD;
}

template<> inline EFeatureType CSimplePreProc<char>::get_feature_type()
{
	return F_CHAR;
}

template<> inline EFeatureType CSimplePreProc<uint8_t>::get_feature_type()
{
	return F_BYTE;
}

template<> inline EFeatureType CSimplePreProc<uint64_t>::get_feature_type()
{
	return F_ULONG;
}
}
#endif
