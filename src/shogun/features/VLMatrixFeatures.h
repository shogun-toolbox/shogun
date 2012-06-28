/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _VL_MATRIX_FEATURES__H__
#define _VL_MATRIX_FEATURES__H__

#include <shogun/features/Features.h>
#include <shogun/lib/SGStringList.h>

namespace shogun
{

/**
 * @brief Class CVLMatrixFeatures used to represent data whose features are
 * better represented with variable length matrices than with unidimensional
 * arrays or vectors. Each of the vectors or examples has the same number of
 * features and these are represented as sequences. The length of the sequences
 * does NOT require to be the same, neither among the features of the same
 * vector nor among features of different vectors.
 */
template< class ST > class CVLMatrixFeatures : public CFeatures
{
	public:
		/** constructor
		 *
		 * @param num_vec number of vectors
		 * @param num_feat number of features per vector
		 */
		CVLMatrixFeatures(int32_t num_vec = 0, int32_t num_feat = 0);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** destructor */
		virtual ~CVLMatrixFeatures();

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
		SGStringList< ST > get_feature_vector(int32_t num) const;

		/** set feature vector num
		 *
		 * @param vec feature vector
		 * @param num index of vector to set
		 */
		void set_feature_vector(SGStringList< ST > const & vec, int32_t num);

		/** get features
		 *
		 * @return features
		 */
		SGStringList< ST >* get_features() const;

		/** set features
		 *
		 * @param features to set
		 * @param num_vec number of vectors
		 */
		void set_features(SGStringList< ST >* features, int32_t num_vec);

		/** @return object name */
		virtual const char* get_name() const { return "VLMatrixFeatures"; }

	private:
		/** internal initialization */
		void init();

	private:
		/** number of vectors or examples */
		int32_t m_num_vectors;

		/** number of features for each vector or example */
		int32_t m_num_features;

		/**
		 * array (better thought as a matrix) of features lists
		 * It constains m_num_vectors lists
		 */
		SGStringList< ST >* m_features;

}; /* class CVLMatrixFeatures */

} /* namespace shogun */

#endif /* _VL_MATRIX_FEATURES__H__ */
