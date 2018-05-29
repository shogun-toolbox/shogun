/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Soumyajit De, Yuyu Zhang, Evan Shelhamer, 
 *          Sergey Lisitsyn
 */

#ifndef _CSPARSEPREPROC__H__
#define _CSPARSEPREPROC__H__

#include <shogun/lib/config.h>

#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/Preprocessor.h>

#include <stdio.h>

namespace shogun
{
template <class ST> class SGSparseVector;
template <class ST> class CSparseFeatures;

/** @brief Template class SparsePreprocessor, base class for preprocessors (cf. CPreprocessor)
 * that apply to CSparseFeatures
 *
 * Two new functions apply_to_sparse_feature_vector() and
 * apply_to_sparse_feature_matrix() are defined in this interface that need to
 * be implemented in each particular preprocessor operating on CSparseFeatures.
 *
 * */
template <class ST> class CSparsePreprocessor : public CPreprocessor
{
public:
	/** constructor
	 */
	CSparsePreprocessor() : CPreprocessor() {}

	/** generic interface for applying the preprocessor. used as a wrapper
	 * for apply_to_sparse_feature_matrix() method
	 *
	 * @param features the sparse input features
	 * @return the result feature object after applying the preprocessor
	 */
	virtual CFeatures* transform(CFeatures* features, bool inplace);

#ifndef SWIG
	[[deprecated]]
#endif
		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGSparseVector<ST>*
		apply_to_sparse_feature_matrix(CSparseFeatures<ST>* f) = 0;

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual SGSparseVector<ST>*
	apply_to_sparse_feature_vector(SGSparseVector<ST>* f, int32_t& len) = 0;

	/// return that we are simple minded features (just fixed size matrices)
	virtual EFeatureClass get_feature_class()
	{
		return C_SPARSE;
	}

	/// return the name of the preprocessor
	virtual const char* get_name() const
	{
		return "UNKNOWN";
	}

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const
	{
		return P_UNKNOWN;
	}

protected:
	virtual SGSparseMatrix<ST>
	apply_to_sparse_matrix(SGSparseMatrix<ST> matrix) = 0;
};

template <class ST>
CFeatures* CSparsePreprocessor<ST>::transform(CFeatures* features, bool inplace)
{
	SG_REF(features);

	auto feature_matrix =
		features->as<CSparseFeatures<ST>>()->get_sparse_feature_matrix();

	if (!inplace)
	{
		// feature_matrix = feature_matrix.clone();
		SG_SERROR("Out-of-place mode for SparsePreprocessor is not supported");
	}

	apply_to_sparse_matrix(feature_matrix);

	auto processed = new CSparseFeatures<ST>(feature_matrix);
	SG_UNREF(features);

	return processed;
}
}
#endif
