/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Vojtech Franc, Soeren Sonnenburg
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2010 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef _LBP_PYR_DOTFEATURES__H__
#define _LBP_PYR_DOTFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Implements Local Binary Patterns with Scale Pyramids as dot features for a set
 * of images. Expects the images to be loaded in a CDenseFeatures object.
 *
 * Original code can be found here : https://github.com/grenaud/freeIbis/tree/master/libocas_v093
 *
 */
class CLBPPyrDotFeatures : public CDotFeatures
{
	public:
		/** default constructor  */
		CLBPPyrDotFeatures();

		/** constructor that initializes this Features object with the images to work on and
		 * the number of pyramids to consider.
		 *
		 * @param image_set the image set
		 * @param image_w image width
		 * @param image_h image height
		 * @param num_pyramids the number of pyramids to consider
		 */
		CLBPPyrDotFeatures(CDenseFeatures<uint32_t>* image_set, int32_t image_w, int32_t image_h,
			uint16_t num_pyramids);

		/** Destructor */
		virtual ~CLBPPyrDotFeatures();

		/** copy constructor
		 *
		 * not implemented!
		 *
		 * @param orig original PolyFeature
		 */
		CLBPPyrDotFeatures(const CLBPPyrDotFeatures & orig);

		/** get dimensions of feature space
		 *
		 * @return dimensions of feature space
		 */
		virtual int32_t get_dim_feature_space() const;

		/** get number of non-zero features in vector
		 *
		 * @param num index of vector
		 * @return number of non-zero features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num);

		/** get feature type
		 *
		 * @return feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		virtual int32_t get_num_vectors() const;

		/** compute dot product between vector1 and vector2,
		 *  appointed by their indices
		 *
		 *  @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index);

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
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator);

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		virtual void free_feature_iterator(void* iterator);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		CFeatures* duplicate() const;

		/**
		 *
		 * @return name of class
		 */
		virtual const char* get_name() const { return "LBPPyrDotFeatures"; }

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
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
				float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** gets a copy of the specified image.
		 *
		 * @param index the index of the image to return
		 * @param width the width of the image (returned by reference)
		 * @param height the height of the image (returned by reference)
		 * @return the image at the given index
		 */
		uint32_t* get_image(int32_t index, int32_t& width, int32_t& height);

		/** returns the transformed representation of the image
		 *
		 * @param index the index of the image
		 */
		SGVector<char> get_transformed_image(int32_t index);
	protected:

		/** lib lbp pyr get dim
		 * @param nPyramids
		 */
		uint32_t liblbp_pyr_get_dim(uint16_t nPyramids);

		/** create the 3x3 local binary pattern with center at (x,y) in image img.
		 *
		 * @param img the image to work on
		 * @param x the x index
		 * @param y the y index
		 */
		uint8_t create_lbp_pattern(uint32_t* img, int32_t x, int32_t y);

	private:

		/** init */
		void init(CDenseFeatures<uint32_t>* image_set, int32_t image_w,
				int32_t image_h);

	protected:
		/** features in original space */
		CDenseFeatures<uint32_t>* images;

		/** image width */
		int32_t image_width;

		/** image height */
		int32_t image_height;

		/** vec nDim */
		int32_t vec_nDim;
};
}
#endif /* _LBP_PYR_DOTFEATURES__H__  */
