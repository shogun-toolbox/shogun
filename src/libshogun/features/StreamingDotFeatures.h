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

#include "lib/common.h"
#include "lib/Time.h"
#include "lib/Mathematics.h"
#include "features/StreamingFeatures.h"
#include "lib/StreamingFile.h"

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

	virtual void init() { };

	virtual void init(CStreamingFile *file, bool is_labelled, int32_t size) { };

public:
	/** Constructor */
	CStreamingDotFeatures()
		: CStreamingFeatures()
	{
		init();
		set_property(FP_DOT);
	}

	/**
	 * Constructor with input information passed.
	 *
	 * @param file CStreamingFile to take input from.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of examples to be held in the parser's "ring".
	 */
	CStreamingDotFeatures(CStreamingFile* file, bool is_labelled, int32_t size)
		: CStreamingFeatures()
	{
		init(file, is_labelled, size);
		set_property(FP_DOT);
	}

	virtual ~CStreamingDotFeatures() { }

	/** compute dot product between vectors of two
	 * StreamingDotFeatures objects.
	 *
	 * @param df StreamingDotFeatures (of same kind) to compute
	 * dot product with
	 */
	virtual float64_t dot(CStreamingDotFeatures* df)=0;

	/** compute dot product between current vector and a dense vector
	 *
	 * @param vec real valued vector of type SGVector
	 */
	virtual float64_t dense_dot(SGVector<float64_t> &vec)=0;

	/** add current vector multiplied with alpha to dense vector, 'vec'
	 *
	 * @param alpha scalar alpha
	 * @param vec real valued vector to add to, encapsulated in an SGVector object
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, SGVector<float64_t> &vec, bool abs_val=false)=0;

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space()=0;

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * @param vector_index the index of the vector over whose components to
	 * 			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator()
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
		return false;
	}

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	virtual void free_feature_iterator(void* iterator)
	{
		SG_NOTIMPLEMENTED;
		return;
	}

protected:

	/// feature weighting in combined dot features
	float64_t combined_weight;
};
}
#endif // _STREAMING_DOTFEATURES__H__
