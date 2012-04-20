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

#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/Preprocessor.h>

namespace shogun
{
template <class ST> class CSimpleFeatures;

/** @brief Template class SimplePreprocessor, base class for preprocessors (cf.
 * CPreprocessor) that apply to CSimpleFeatures (i.e. rectangular dense matrices)
 *
 * Two new functions apply_to_feature_vector() and apply_to_feature_matrix()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CSimpleFeatures. For examples see e.g. CLogPlusOne
 * or CPCACut.
 */
template <class ST> class CSimplePreprocessor : public CPreprocessor
{
	public:
		/** constructor
		 */
		CSimplePreprocessor();

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<ST> apply_to_feature_matrix(CFeatures* features)=0;

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<ST> apply_to_feature_vector(const SGVector<ST>& vector)=0;

		/// return that we are simple features (just fixed size matrices)
		virtual EFeatureClass get_feature_class();
		/// return feature type
		virtual EFeatureType get_feature_type();

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const;

};

}
#endif
