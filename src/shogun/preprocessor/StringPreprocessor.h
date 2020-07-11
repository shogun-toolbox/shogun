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
template <class ST> class StringFeatures;

/** @brief Template class StringPreprocessor, base class for preprocessors (cf.
 * Preprocessor) that apply to CStringFeatures (i.e. strings of variable
 * length).
 *
 * Two new functions apply_to_string() and apply_to_string_list()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CStringFeatures.
 */
template <class ST> class StringPreprocessor : public Preprocessor
{
	public:
		/** constructor
		 */
		StringPreprocessor() : Preprocessor() {}

		/** Apply transformation to string features.
		 *
		 * @param features the string input features
		 * @return the result feature object after applying the preprocessor
		 */
		std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true) override;

		/// apply preproc on single feature vector
		virtual ST* apply_to_string(ST* f, int32_t &len)=0;

		/// return that we are string features (just fixed size matrices)
		EFeatureClass get_feature_class() override { return C_STRING; }
		/// return feature type
		EFeatureType get_feature_type() override;

		/// return the name of the preprocessor
		const char* get_name() const override { return "UNKNOWN"; }

		/// return a type of preprocessor
		EPreprocessorType get_type() const override { return P_UNKNOWN; }

	protected:
		/** apply the preprocessor to string list in place.
		 *
		 * @param string_list the string list to be preprocessed
		 */
		virtual void apply_to_string_list(std::vector<SGVector<ST>>& string_list) = 0;
};

}
#endif
