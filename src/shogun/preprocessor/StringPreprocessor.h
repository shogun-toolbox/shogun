/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Soumyajit De, Yuyu Zhang, Sergey Lisitsyn
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

		/** generic interface for applying the preprocessor. used as a wrapper
		 * for apply_to_string_features() method
		 *
		 * @param features the string input features
		 * @return the result feature object after applying the preprocessor
		 */
		virtual CFeatures* apply(CFeatures* features, bool inplace = true);

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

}
#endif
