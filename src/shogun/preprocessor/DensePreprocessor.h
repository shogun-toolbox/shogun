/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Soumyajit De, Sergey Lisitsyn
 */

#ifndef _DENSEPREPROCESSOR__H__
#define _DENSEPREPROCESSOR__H__

#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/Preprocessor.h>

namespace shogun
{
template <class ST> class CDenseFeatures;

/** @brief Template class DensePreprocessor, base class for preprocessors (cf.
 * CPreprocessor) that apply to CDenseFeatures (i.e. rectangular dense matrices)
 *
 * Two new functions apply_to_feature_vector() and apply_to_feature_matrix()
 * are defined in this interface that need to be implemented in each particular
 * preprocessor operating on CDenseFeatures. For examples see e.g. CLogPlusOne
 * or CPCACut.
 */
template <class ST> class CDensePreprocessor : public CPreprocessor
{
	public:
		/** constructor
		 */
		CDensePreprocessor();

		/** Apply transformation to dense features.
		 *
		 * @param features the dense input features
		 * @return the result feature object after applying the preprocessor
		 */
		virtual CFeatures* transform(CFeatures* features, bool inplace = true);

#ifndef SWIG
		[[deprecated]]
#endif
		    /// apply preproc on feature matrix
		    /// result in feature matrix
		    /// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		    // remove after cleaning up codebase
		    virtual SGMatrix<ST>
		    apply_to_feature_matrix(CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<ST> apply_to_feature_vector(SGVector<ST> vector) = 0;

		/// return that we are dense features (just fixed size matrices)
		virtual EFeatureClass get_feature_class();
		/// return feature type
		virtual EFeatureType get_feature_type();

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const;

	protected:
		/** Apply preprocessor on matrix. Subclasses should try to apply in
		 * place to avoid copying.
		 * @param matrix the input feature matrix
		 * @return the matrix after applying the preprocessor
		 */
		virtual SGMatrix<ST> apply_to_matrix(SGMatrix<ST> matrix) = 0;
};

}
#endif
