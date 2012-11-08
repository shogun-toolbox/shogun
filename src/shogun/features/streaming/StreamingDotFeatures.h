/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_DOTFEATURES__H__
#define _STREAMING_DOTFEATURES__H__

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/streaming/StreamingFile.h>

namespace shogun
{
/** @brief Streaming features that support dot products among other operations.
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

class CStreamingDotFeatures : public CStreamingFeatures
{

public:
	/** Constructor */
	CStreamingDotFeatures();

	/**
	 * Constructor with input information passed.
	 *
	 * @param file CStreamingFile to take input from.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of examples to be held in the parser's "ring".
	 */
	CStreamingDotFeatures(CStreamingFile* file, bool is_labelled, int32_t size);

	/**
	 * Constructor taking a CDotFeatures object and optionally,
	 * labels, as args.
	 *
	 * The derived class should implement it so that the
	 * Streaming*Features class uses the DotFeatures object as the
	 * input, getting examples one by one from the DotFeatures
	 * object (and labels, if applicable).
	 *
	 * @param dot_features CDotFeatures object
	 * @param lab labels (optional)
	 */
	CStreamingDotFeatures(CDotFeatures* dot_features, float64_t* lab=NULL);

	virtual ~CStreamingDotFeatures();

	/** compute dot product between vectors of two
	 * StreamingDotFeatures objects.
	 *
	 * @param df StreamingDotFeatures (of same kind) to compute
	 * dot product with
	 */
	virtual float32_t dot(CStreamingDotFeatures* df)=0;

	/** compute dot product between current vector and a dense vector
	 *
	 * @param vec2 real valued vector
	 * @param vec2_len length of vector
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len)=0;

	/** Compute the dot product for all vectors. This function makes use of dense_dot
	 * alphas[i] * sparse[i]^T * w + b
	 *
	 * @param output result for the given vector range
	 * @param alphas scalars to multiply with, may be NULL
	 * @param vec dense vector to compute dot product with
	 * @param dim length of the dense vector
	 * @param b bias
	 * @param num_vec number of vectors to operate on (indices 0 to num_vec-1)
	 *
	 * If num_vec == 0 or left to its default value, the function
	 * attempts to return dot product for all vectors.  However,
	 * the given output vector must be preallocated!
	 *
	 * note that the result will be written to output[0...(num_vec-1)]
	 * except when num_vec = 0
	 */
	virtual void dense_dot_range(float32_t* output, float32_t* alphas,
			float32_t* vec, int32_t dim, float32_t b, int32_t num_vec=0);

	/** add current vector multiplied with alpha to dense vector, 'vec'
	 *
	 * @param alpha scalar alpha
	 * @param vec2 real valued vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val=false)=0;

	/**
	 * Expand the vector passed so that it its length is equal to
	 * the dimensionality of the features. The previous values are
	 * kept intact through realloc, and the new ones are set to zero.
	 *
	 * @param vec float32_t* vector
	 * @param len length of the vector
	 */
	virtual void expand_if_required(float32_t*& vec, int32_t &len);

	/**
	 * Expand the vector passed so that it its length is equal to
	 * the dimensionality of the features. The previous values are
	 * kept intact through realloc, and the new ones are set to zero.
	 *
	 * @param vec float64_t* vector
	 * @param len length of the vector
	 */
	virtual void expand_if_required(float64_t*& vec, int32_t &len);

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const=0;

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator();

	/** get number of non-zero features in vector
	 *
	 * (in case accurate estimates are too expensive overestimating is OK)
	 *
	 * @return number of sparse features in vector
	 */
	virtual int32_t get_nnz_features_for_vector();

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
	virtual bool get_next_feature(int32_t& index, float32_t& value, void* iterator);

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	virtual void free_feature_iterator(void* iterator);

protected:

	/// feature weighting in combined dot features
	float32_t combined_weight;
};
}
#endif // _STREAMING_DOTFEATURES__H__
