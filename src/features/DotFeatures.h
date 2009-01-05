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
	virtual float64_t add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len)=0;

	/** Compute the dot product for a range of vectors. This function makes use of dense_dot
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
};
#endif // _DOTFEATURES_H___
