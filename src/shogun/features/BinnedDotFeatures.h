/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef _BINNED_DOTFEATURES_H___
#define _BINNED_DOTFEATURES_H___

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
	template <class T> class CDenseFeatures;

/** @brief The class BinnedDotFeatures contains a 0-1 conversion of features into bins.
 *
 * It is often useful to convert real valued features into 0-1 vectors by
 * defining a fixed set of bins and then filling all bins with 0 except the
 * bin into which the value falls, i.e.bin=1 iff
 * bin lower limit <= value < bin upper limit
 *
 * This class optionally allows to fill all bins with 1 up to the one with
 * value < bin upper limit when the fill flag is set.
 *
 * In addition one may choose to normalize vectors to have norm one (if the
 * norm one flag is set).
 *
 * Bins take the form of a matrix with as many columns as there are input
 * dimensions. Then each row vector contains the limits of the bins.
 *
 * Note that BinnedDotFeatures never *explicitly* compute the binned feature
 * representation but only overload the abstract dot/add methods in
 * CDotFeatures making them highly memory and computationally efficient.
 */
class CBinnedDotFeatures : public CDotFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CBinnedDotFeatures(int32_t size=0);

		/** copy constructor */
		CBinnedDotFeatures(const CBinnedDotFeatures & orig);

		/** constructor
		 *
		 * @param sf CSimpleFeatureMatrix of type float64_t to convert into
		 * binned features
		 * @param bins a matrix with bins to compute binned features from
		 */
		CBinnedDotFeatures(CDenseFeatures<float64_t>* sf, SGMatrix<float64_t> bins);

		virtual ~CBinnedDotFeatures();

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const;

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** get number of non-zero features in vector
		 *
		 * (in case accurate estimates are too expensive overestimating is OK)
		 *
		 * @param num which vector
		 * @return number of sparse features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num);

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


		/** get the fill flag
		 *
		 * @return fill flag - if true bins are filled up to value v
		 */
		bool get_fill();

		/** set the fill flag
		 *
		 * @param fill - if fill is true bins are filled up to value v
		 */
		void set_fill(bool fill);

		/** get norm one flag
		 *
		 * @return norm one flag - if true vectors are normalized to have norm one
		 */
		bool get_norm_one();

		/** set norm one flag
		 *
		 * @param norm_one if norm_one is true vectors are normalized to have norm one
		 */
		void set_norm_one(bool norm_one);

		/** set features to convert to binned features
		 *
		 * @param features - features to convert to binned features
		 */
		void set_simple_features(CDenseFeatures<float64_t>* features);

		/** get features that are convert to binned features
		 *
		 * @return features - simple features object
		 */
		CDenseFeatures<float64_t>* get_simple_features();

		/** set bins
		 *
		 * bins have a matrix shape with as many column vectors as there are
		 * dimensions in the corresponding feature object. The column vector
		 * then contains the limits, e.g. linspace from minimum to maximum
		 * value of that dimension.
		 *
		 * @param bins - bins to convert features into
		 */
		void set_bins(SGMatrix<float64_t> bins);

		/** get current bins
		 *
		 * @return bins
		 */
		SGMatrix<float64_t> get_bins();

		/** Returns the name of the Object.
		 *  @return name "BinnedDotFeatures"
		 */
		virtual const char* get_name() const;

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

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

		/** get number of examples/vectors
		 *
		 * @return number of examples/vectors
		 */
		virtual int32_t get_num_vectors() const;

	private:
		void init();

		/** test if feature matrix matches size of bins with limits
		 *
		 * @param vec2_len length of dense vector
		 */
		void assert_shape(int32_t vec2_len);

	protected:
		/// underlying features
		CDenseFeatures<float64_t>* m_features;

		/// bins with limits
		SGMatrix<float64_t> m_bins;

		/// fill up with 1's or flag just one column
		bool m_fill;

		/// normalize vectors to have norm one
		bool m_norm_one;
};
}
#endif // _BINNED_DOTFEATURES_H___

