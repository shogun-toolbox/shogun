/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
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

	/// apply preproc on feature matrix
	/// result in feature matrix
	/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
	virtual SGSparseVector<ST>* apply_to_sparse_feature_matrix(CSparseFeatures<ST>* f)=0;

	/// apply preproc on single feature vector
	/// result in feature matrix
	virtual SGSparseVector<ST>* apply_to_sparse_feature_vector(SGSparseVector<ST>* f, int32_t &len)=0;

	/// return that we are simple minded features (just fixed size matrices)
	virtual EFeatureClass get_feature_class() { return C_SPARSE; }

	/// return the name of the preprocessor
	virtual const char* get_name() const { return "UNKNOWN"; }

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const { return P_UNKNOWN; }

};
}
#endif
