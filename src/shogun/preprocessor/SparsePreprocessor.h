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
template <class ST> class SparseFeatures;

/** @brief Template class SparsePreprocessor, base class for preprocessors (cf.
 * Preprocessor) that apply to SparseFeatures
 *
 * Two new functions apply_to_sparse_feature_vector() and
 * apply_to_sparse_matrix() are defined in this interface that need to
 * be implemented in each particular preprocessor operating on SparseFeatures.
 *
 * */
template <class ST> class SparsePreprocessor : public Preprocessor
{
public:
	/** constructor
	 */
	SparsePreprocessor() : Preprocessor() {}

	/** generic interface for applying the preprocessor. used as a wrapper
	 * for apply_to_sparse_feature_matrix() method
	 *
	 * @param features the sparse input features
	 * @return the result feature object after applying the preprocessor
	 */
	virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace);

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
std::shared_ptr<Features> SparsePreprocessor<ST>::transform(std::shared_ptr<Features> features, bool inplace)
{
	

	auto feature_matrix =
		features->as<SparseFeatures<ST>>()->get_sparse_feature_matrix();

	if (!inplace)
	{
		// feature_matrix = feature_matrix.clone();
		error("Out-of-place mode for SparsePreprocessor is not supported");
	}

	apply_to_sparse_matrix(feature_matrix);

	auto processed = new SparseFeatures<ST>(feature_matrix);
	

	return processed;
}
}
#endif
