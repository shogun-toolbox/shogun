/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DOTFEATURES_H___
#define _DOTFEATURES_H___

#include "lib/common.h"
#include "features/Features.h"

/** Features that support the following operations:
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
 * - multiplication with a scalar \f$\alpha\f$ and addition on to a dense vector \f${\bf z}\f$:
 *
 *   \f[${\bf z'} = \alpha {\bf x} + {\bf z}\f]
 * 
 */
class CDotFeatures : public CFeatures
{
	public:

		/** constructor
		 *
		 * @param size cache size
		 */
		CDotFeatures(int32_t size=0) : CFeatures(size), combined_weight(1.0)
		{
			set_property(FP_DOT);
		}

		/** copy constructor */
		CDotFeatures(const CDotFeatures & orig) :
			CFeatures(orig), combined_weight(orig.combined_weight)  {}

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CDotFeatures(char* fname) : CFeatures(fname) {}

		virtual ~CDotFeatures() { }

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space()=0;

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, int32_t vec_idx2)=0;

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
		 */
		virtual void dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b);

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

	protected:
		inline void display_progress(int32_t start, int32_t stop, int32_t v)
		{
			int32_t num_vectors=stop-start;
			int32_t i=v-start;

			if ( (i% (num_vectors/100+1))== 0)
				SG_PROGRESS(v, 0.0, num_vectors-1);
		}
	protected:
		float64_t combined_weight;
};
#endif // _DOTFEATURES_H___
