/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Vojtech Franc, Soeren Sonnenburg
 * Copyright (C) 2010 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef _LBP_PYR_DOTFEATURES__H__
#define _LBP_PYR_DOTFEATURES__H__

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SimpleFeatures.h>

namespace shogun
{
/** @brief implement DotFeatures for the polynomial kernel
 *
 * see DotFeatures for further discription
 *
 */
class CLBPPyrDotFeatures : public CDotFeatures
{
	public:
		/** default constructor  */
		CLBPPyrDotFeatures(void);

		/** constructor
		 * 
		 * @param images images
		 */
		CLBPPyrDotFeatures(CSimpleFeatures<uint32_t>* images, uint16_t num_pyramids);

		virtual ~CLBPPyrDotFeatures();

		/** copy constructor
		 * 
		 * not implemented!
		 *
		 * @param orig original PolyFeature
		 */ 
		CLBPPyrDotFeatures(const CLBPPyrDotFeatures & orig){ 
			SG_PRINT("CLBPPyrDotFeatures:\n");
			SG_NOTIMPLEMENTED;};

		/** get dimensions of feature space
		 *
		 * @return dimensions of feature space
		 */ 
		inline virtual int32_t get_dim_feature_space() const
		{
			return vec_nDim;
		}

		/** get number of non-zero features in vector
		 *
		 * @param num index of vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			return vec_nDim;
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
		inline virtual int32_t get_num_vectors() const
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
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

		/**
		 *
		 * @return size
		 */
		inline virtual int32_t get_size()
		{
			return sizeof(float64_t);
		}

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 * 			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index)
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_first_feature
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_first_feature
		 * @return true if a new non-zero feature got returned
		 */
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		virtual void free_feature_iterator(void* iterator)
		{
			SG_NOTIMPLEMENTED;
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
		inline virtual const char* get_name() const { return "LBPPyrDotFeatures"; }

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
		
		/** lib lbp pyr get dim
		 * @param nPyramids
		 */
		uint32_t liblbp_pyr_get_dim(uint16_t nPyramids);

	protected:
		/** features in original space*/
		CSimpleFeatures<uint32_t>* m_feat;

		/** img */
		uint32_t* img;
		/** img nRows */
		int32_t img_nRows;
		/** img nCols */
		int32_t img_nCols;
		/** vec nDim */
		int32_t vec_nDim;
};
}
#endif /* _LBP_PYR_DOTFEATURES__H__  */
