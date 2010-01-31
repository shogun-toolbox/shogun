/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef _SPARSEPOLYFEATURES__H__
#define _SPARSEPOLYFEATURES__H__

#include "lib/common.h"
#include "features/DotFeatures.h"
#include "features/SparseFeatures.h"

namespace shogun
{
/** @brief implement DotFeatures for the polynomial kernel
 *
 * see DotFeatures for further discription
 *
 */
class CSparsePolyFeatures : public CDotFeatures
{
	public:
		/** constructor
		 * 
		 * @param feat real features
		 * @param degree degree of the polynomial kernel
		 * 					(only degree 2 & 3 are supported)
		 * @param normalize normalize kernel
		 */
		CSparsePolyFeatures(CSparseFeatures<float64_t>* feat, int32_t degree, bool normalize, int32_t hash_bits);

		virtual ~CSparsePolyFeatures();

		/** copy constructor
		 * 
		 * not implemented!
		 *
		 * @param orig original PolyFeature
		 */ 
		CSparsePolyFeatures(const CSparsePolyFeatures & orig){ 
			SG_PRINT("CSparsePolyFeatures:\n");
			SG_NOTIMPLEMENTED;};

		/** get dimensions of feature space
		 *
		 * @return dimensions of feature space
		 */ 
		inline virtual int32_t get_dim_feature_space()
		{
			return m_output_dimensions;
		}

		/** get number of non-zero features in vector
		 *
		 * @param num index of vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			return m_output_dimensions;
		}

		/** get feature type
		 *
		 * @return feature type
		 */
		inline virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class
		 */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_POLY;
		}

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		inline virtual int32_t get_num_vectors()
		{
			if (m_feat)
				return m_feat->get_num_vectors();
			else
				return 0;

		}

		/** compute dot product between vector1 and vector2,
		 *  appointed by their indices
		 *
		 *  @param vec_idx1 index of first vector
		 *   @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, int32_t vec_idx2);

		/**
		 *
		 * @return size
		 */
		inline virtual int32_t get_size()
		{
			return sizeof(float64_t);
		}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		CFeatures* duplicate() const;

		/**
		 *
		 * @return name of class
		 */
		inline virtual const char* get_name() const { return "SparsePolyFeatures"; }

		/** compute dot product of vector with index arg1 
		 *  with an given second vector 
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 second vector
		 * @param vec2_len length of second vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

		/** compute alpha*x+vec2
		 * 
		 * @param alpha alpha
		 * @param vec_idx1 index of first vector x
		 * @param vec2 vec2
		 * @param vec2_len length of vec2
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

	protected: 
		/** store the norm of each training example */
		void store_normalization_values();

	protected:
		/** features in original space*/
		CSparseFeatures<float64_t>* m_feat;
		/** degree of the polynomial kernel */
		int32_t m_degree;
		/** normalize */
		bool m_normalize;
		/** dimensions of the input space */
		int32_t m_input_dimensions;
		/** dimensions of the feature space of the polynomial kernel */
		int32_t m_output_dimensions;
		/**store norm of each training example */
		float64_t* m_normalization_values; 
		/** mask */
		uint32_t mask;
		/** number of bits in hash */
		int32_t m_hash_bits;
};
}
#endif // _SPARSEPOLYFEATURES__H__
