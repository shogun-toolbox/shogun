/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _MATRIX_FEATURES__H__
#define _MATRIX_FEATURES__H__

#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrixList.h>


namespace shogun
{

/**
 * @brief Class CMatrixFeatures used to represent data whose feature vectors are
 * better represented with matrices rather than with unidimensional arrays or
 * vectors. Optionally, it can be restricted that all the feature vectors have
 * the same number of features. Set the attribute num_features different to zero
 * to use this restriction. Allow feature vectors with different number of
 * features by setting num_features equal to zero (default behaviour).
 */
template< class ST > class CMatrixFeatures : public CFeatures
{
	public:
		/** default constructor */
		CMatrixFeatures();

		/** standard constructor
		 *
		 * @param num_vecs number of vectors
		 * @param num_feats number of features per vector
		 */
		CMatrixFeatures(int32_t num_vecs, int32_t num_feats = 0);

		/** constructor
		 *
		 * @param feats list of feature matrices
		 * @param num_feats number of features per vector
		 */
		CMatrixFeatures(SGMatrixList< ST > feats, int32_t num_feats = 0);

		/**
		 * constructor using the data of all the features concatenated
		 * in a matrix. All the features are assumed to have the same
		 * length. The number of colums of feats must be equal to
		 * feat_length times num_vecs. The number of features per vector
		 * is equal to the number of rows of feats.
		 *
		 * @param feats concatenation of the features
		 * @param feat_length length of each feature
		 * @param num_vecs number of feature vectors
		 */
		CMatrixFeatures(SGMatrix< ST > feats, int32_t feat_length, int32_t num_vecs);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** destructor */
		virtual ~CMatrixFeatures();

		/** get feature type
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class
		 *
		 * @return feature class like STRING, SIMPLE, SPARSE... (C_MATRIX in this case)
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of examples/vectors, possibly corresponding to
		 * the current subset
		 *
		 * @return number of examples/vectors (possibly of subset, if
		 * implemented)
		 */
		virtual int32_t get_num_vectors() const { return m_num_vectors; }

		/** get feature vector num
		 *
		 * @param num feature vector index
		 *
		 * @return feature vector
		 */
		SGMatrix< ST > get_feature_vector(int32_t num) const;

		/** get a column of a feature vector
		 *
		 * @param out where the column will be copied
		 * @param num index of the feature vector
		 * @param col index of the column to get
		 */
		void get_feature_vector_col(SGVector< ST > out, int32_t num, int32_t col) const;

		/** set feature vector num
		 *
		 * @param vec feature vector
		 * @param num index of vector to set
		 */
		void set_feature_vector(SGMatrix< ST > const vec, int32_t num);

		/** get features
		 *
		 * @return features
		 */
		inline SGMatrixList< ST > get_features() const { return m_features; }

		/** set features
		 *
		 * @param features to set
		 * @param num_feats number of features per vector
		 */
		void set_features(SGMatrixList< ST > features, int32_t num_feats);

		/** @return object name */
		virtual const char* get_name() const { return "MatrixFeatures"; }

		/** @return the number of features */
		inline int32_t get_num_features() const { return m_num_features; }

		/** helper method used to specialize a base class instance
		 *
		 */
		static CMatrixFeatures* obtain_from_generic(CFeatures* const base_features);

	private:
		/** internal initialization */
		void init();

		/** cleanup matrix features */
		void cleanup();

	private:
		/** number of vectors or examples */
		int32_t m_num_vectors;

		/** number of features for each vector or example */
		int32_t m_num_features;

		/** list of m_num_vectors matrices (the so-called feature vectors) */
		SGMatrixList< ST > m_features;

}; /* class CMatrixFeatures */

} /* namespace shogun */

#endif /* _MATRIX_FEATURES__H__ */
