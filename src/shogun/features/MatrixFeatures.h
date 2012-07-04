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
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/**
 * @brief Class CMatrixFeatures used to represent data whose features are
 * better represented with variable length matrices than with unidimensional
 * arrays or vectors. Each of the vectors or examples has the same number of
 * features and these are represented as sequences. The length of the sequences
 * does NOT require to be the same, neither among the features of the same
 * vector nor among features of different vectors.
 */
template< class ST > class CMatrixFeatures : public CFeatures
{
	public:
		/** default constructor */
		CMatrixFeatures();

		/** standard constructor
		 *
		 * @param num_vec number of vectors
		 * @param num_feat number of features per vector
		 */
		CMatrixFeatures(int32_t num_vec, int32_t num_feat);

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
		 * @return feature class like STRING, SIMPLE, SPARSE...
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of examples/vectors, possibly corresponding to
		 * the current subset
		 *
		 * @return number of examples/vectors (possibly of subset, if
		 * implemented)
		 */
		virtual int32_t get_num_vectors() const { return m_num_vectors; }

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size() const;

		/** get feature vector num
		 *
		 * @param num feature vector index
		 *
		 * @return feature vector
		 */
		SGMatrix< ST > get_feature_vector(int32_t num) const;

		/** TODO doc */
		void get_feature_vector_col(SGVector< ST > out, int32_t num, int32_t col) const;

		/** set feature vector num
		 *
		 * @param vec feature vector
		 * @param num index of vector to set
		 */
		void set_feature_vector(SGMatrix< ST > const & vec, int32_t num);

		/** get features
		 *
		 * @param num_vec number of feature vectors
		 *
		 * @return features
		 */
		SGMatrix< ST >* get_features(int32_t& num_vec) const;

		/** set features
		 *
		 * @param features to set
		 * @param num_vec number of vectors
		 */
		void set_features(SGMatrix< ST >* features, int32_t num_vec);

		/** @return object name */
		virtual const char* get_name() const { return "MatrixFeatures"; }

		/** @return the number of features */
		inline int32_t get_num_features() const { return m_num_features; }

	private:
		/** internal initialization */
		void init();

		/** cleanup matrix features */
		void cleanup();

		/** cleanup multiple feature vectors
		 *
		 * @param start index of first vector to be cleaned
		 * @param stop index of the last vector to be cleaned
		 * */
		virtual void cleanup_feature_vectors(int32_t start, int32_t stop);

	private:
		/** number of vectors or examples */
		int32_t m_num_vectors;

		/** number of features for each vector or example */
		int32_t m_num_features;

		/** list of m_num_vectors matrices (the so-called feature vectors) */
		SGMatrix< ST >* m_features;

}; /* class CMatrixFeatures */

} /* namespace shogun */

#endif /* _MATRIX_FEATURES__H__ */
