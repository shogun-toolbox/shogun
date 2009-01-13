/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPECFEATURES_H___
#define _SPECFEATURES_H___

#include "lib/common.h"
#include "lib/io.h"
#include "features/DotFeatures.h"
#include "features/StringFeatures.h"

template <class ST> class CStringFeatures;

class CSpecFeatures : public CDotFeatures
{
	public:

		/** constructor
		 *
		 * @param str stringfeatures (of words)
		 */
		CSpecFeatures(CStringFeatures<uint16_t>* str);
		virtual ~CSpecFeatures();

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		inline virtual int32_t get_dim_feature_space()
		{
			return w_dim;
		}

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, int32_t vec_idx2);

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
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			SG_NOTIMPLEMENTED;
			return 0;
		}

	protected:
		virtual void obtain_kmer_spectrum();
		virtual void delete_kmer_spectrum();

	protected:
		/** number of strings */
		int32_t num_strings;
		/** strings */
		CStringFeatures<uint16_t>* strings;

		/** degree */
		int32_t degree;
		/** from degree */
		int32_t from_degree;
		/** size of alphabet */
		int32_t alphabet_size;
		/** w dim */
		int32_t w_dim;

		/** size of k-mer spectrum*/
		int32_t spec_size;
		/** k-mer counts for all strings */
		int32_t** k_spectrum;
};
#endif // _SPECFEATURES_H___
