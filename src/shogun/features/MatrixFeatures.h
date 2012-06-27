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
#include <shogun/lib/SGNDArray.h>

namespace shogun
{

/**
 * @brief Class CMatrixFeatures to represent data whose features are better represented
 * with matrices than with unidimensional arrays or vectors.
 */
template< class ST > class CMatrixFeatures : public CFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CMatrixFeatures(int32_t size = 0);

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

		/** get number of examples/vectors, possibly corresponding to the current subset
		 *
		 * @return number of examples/vectors (possibly of subset, if implemented)
		 */
		virtual int32_t get_num_vectors() const { return m_num_vectors; };

		/** get memory footprint of one feature
		 *
		 * abstract base method
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size() const;

		/** get feature matrix num
		 *
		 * @param num index example
		 *
		 * @return feature matrix
		 */
		SGMatrix< ST > get_feature_matrix(int32_t num) const;

		/** set feature matrix num
		 *
		 * @param matrix feature matrix
		 * @param num index of example to set
		 */
		void set_feature_matrix(SGMatrix< ST > matrix, int32_t num);

		/** get feature array
		 *
		 * @return three dimensional feature array
		 */
		SGNDArray< ST > get_feature_array() const;

		/** set feature array
		 *
		 * @param array three dimensional feature array to set
		 */
		void set_feature_array(SGNDArray< ST > array);

		/** @return object name */
		virtual const char* get_name() const { return "MatrixFeatures"; }

	private:
		/** internal initialization */
		void init();

	private:
		/** number of vectors or examples */
		int32_t m_num_vectors;

		/** number of features for each vector or example*/
		int32_t m_num_features;

		/** three dimensional feature array */
		SGNDArray< ST > m_feature_array;

}; /* class CMatrixFeatures */

} /* namespace shogun */

#endif /* _MATRIX_FEATURES__H__ */
