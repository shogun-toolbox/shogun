/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WDFEATURES_H___
#define _WDFEATURES_H___

#include "lib/common.h"
#include "features/DotFeatures.h"
#include "features/StringFeatures.h"

template <class ST> class CStringFeatures;

class CWDFeatures : public CDotFeatures
{
	public:

		/** constructor
		 *
		 * @param str stringfeatures (of bytes)
		 */
		CWDFeatures(CStringFeatures<uint8_t>* str);
		virtual ~CWDFeatures();

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
			int32_t dim=0;
			for (int32_t i=0; i<degree; i++)
				dim+=w_offsets[i]/alphabet_size;

			return dim;
		}


	protected:
		/** set normalization constant */
		void set_normalization_const();

		int32_t set_wd_weights();

	protected:
		CStringFeatures<uint8_t>* strings;

		/** normalization const */
		float64_t normalization_const;

		/** degree */
		int32_t degree;
		/** from degree */
		int32_t from_degree;
		/** length of string in vector */
		int32_t string_length;
		/** size of alphabet */
		int32_t alphabet_size;
		/** w offsets */
		int32_t* w_offsets;
		/** w dim */
		int32_t w_dim;
		/** wd weights */
		float64_t* wd_weights;
};
#endif // _WDFEATURES_H___
