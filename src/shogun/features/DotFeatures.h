/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009-2010 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _DOTFEATURES_H___
#define _DOTFEATURES_H___

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
/** @brief Features that support dot products among other operations.
 *
 * DotFeatures support the following operations:
 *
 * - a way to obtain the dimensionality of the feature space, i.e. \f$\mbox{dim}({\cal X})\f$
 *
 * - dot product between feature vectors:
 *
 *   \f[r = {\bf x} \cdot {\bf x'}\f]
 *
 * - dot product between feature vector and a dense vector \f${\bf z}\f$:
 *
 *   \f[r = {\bf x} \cdot {\bf z}\f]
 *
 * - multiplication with a scalar \f$\alpha\f$ and addition to a dense vector \f${\bf z}\f$:
 *
 *   \f[ {\bf z'} = \alpha {\bf x} + {\bf z} \f]
 *
 * - iteration over all (potentially) non-zero features of \f${\bf x}\f$
 *
 */
class CDotFeatures : public CFeatures
{
	public:

		/** constructor
		 *
		 * @param size cache size
		 */
		CDotFeatures(int32_t size=0);

		/** copy constructor */
		CDotFeatures(const CDotFeatures & orig);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CDotFeatures(CFile* loader);

		virtual ~CDotFeatures() { }

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const=0;

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)=0;

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 dense vector
		 */
		virtual float64_t dense_dot_sgvec(int32_t vec_idx1, const SGVector<float64_t> vec2);

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)=0;

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false)=0;

		/** Compute the dot product for a range of vectors. This function makes use of dense_dot
		 * alphas[i] * sparse[i]^T * w + b
		 *
		 * @param output result for the given vector range
		 * @param start start vector range from this idx
		 * @param stop stop vector range at this idx
		 * @param alphas scalars to multiply with, may be NULL
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 *
		 * note that the result will be written to output[0...(stop-start-1)]
		 */
		virtual void dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b);

		/** Compute the dot product for a subset of vectors. This function makes use of dense_dot
		 * alphas[i] * sparse[i]^T * w + b
		 *
		 * @param sub_index index for which to compute outputs
		 * @param num length of index
		 * @param output result for the given vector range
		 * @param alphas scalars to multiply with, may be NULL
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 */
		virtual void dense_dot_range_subset(int32_t* sub_index, int32_t num,
				float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b);

		/** Compute the dot product for a range of vectors. This function is
		 * called by the threads created in dense_dot_range */
		static void* dense_dot_range_helper(void* p);

		/** get number of non-zero features in vector
		 *
		 * (in case accurate estimates are too expensive overestimating is OK)
		 *
		 * @param num which vector
		 * @return number of sparse features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num)=0;

		/** get combined feature weight
		 *
		 * @return combined feature weight
		 */
		inline float64_t get_combined_feature_weight() { return combined_weight; }

		/** set combined kernel weight
		 *
		 * @param nw new combined feature weight
		 */
		inline void set_combined_feature_weight(float64_t nw) { combined_weight=nw; }

		/** compute the feature matrix in feature space
		 *
		 * @return computed feature matrix
		 */
		SGMatrix<float64_t> get_computed_dot_feature_matrix();

		/** compute the feature vector in feature space
		 *
		 * @return computed feature vector
		 */
		SGVector<float64_t> get_computed_dot_feature_vector(int32_t num);

		/** run benchmark for add_to_dense_vec */
		void benchmark_add_to_dense_vector(int32_t repeats=5);

		/** run benchmark for dense_dot_range */
		void benchmark_dense_dot_range(int32_t repeats=5);

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index)=0;

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_feature_iterator
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_feature_iterator
		 * @return true if a new non-zero feature got returned
		 */
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)=0;

		/** clean up iterator
		 * call this function with the iterator returned by get_feature_iterator
		 *
		 * @param iterator as returned by get_feature_iterator
		 */
		virtual void free_feature_iterator(void* iterator)=0;

		/** get mean
		 *
		 * @return mean returned
		 */
		virtual SGVector<float64_t> get_mean();

		/** get mean of two CDotFeature objects
		 *
		 * @return mean returned
		 */
		static SGVector<float64_t> get_mean(CDotFeatures* lhs, CDotFeatures* rhs);

		/** get covariance
		 *
		 * @return covariance
		 */
		virtual SGMatrix<float64_t> get_cov();

		/** compute the covariance of two CDotFeatures together
		 *
		 * @return covariance
		 */
		static SGMatrix<float64_t> compute_cov(CDotFeatures* lhs, CDotFeatures* rhs);

	protected:
		/** display progress output
		 *
		 * @param start minimum value
		 * @param stop maximum value
		 * @param v current value
		 */
		void display_progress(int32_t start, int32_t stop, int32_t v);

	private:
		void init();

	protected:

		/// feature weighting in combined dot features
		float64_t combined_weight;
};
}
#endif // _DOTFEATURES_H___
