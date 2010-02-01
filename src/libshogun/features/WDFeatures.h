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

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief Features that compute the Weighted Degreee Kernel feature space
 * explicitly.
 *
 * \sa CWeightedDegreeStringKernel
 */
class CWDFeatures : public CDotFeatures
{
	public:

		/** constructor
		 *
		 * @param str stringfeatures (of bytes)
		 * @param order of wd kernel
		 * @param from_order use first order weights from higher order weighting
		 */
		CWDFeatures(CStringFeatures<uint8_t>* str, int32_t order, int32_t from_order);

		/** copy constructor */
		CWDFeatures(const CWDFeatures & orig);

		/** destructor */
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
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			int32_t vlen=-1;
			bool free_vec;
			uint8_t* vec=strings->get_feature_vector(num, vlen, free_vec);
			strings->free_feature_vector(vec, num, free_vec);
			return degree*vlen;
		}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** get feature type
		 *
		 * @return templated feature type
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
			return C_WD;
		}

		inline virtual int32_t get_num_vectors()
		{
			return num_strings;
		}

		inline virtual int32_t get_size()
		{
			return sizeof(float64_t);
		}

		/** set normalization constant
		 * @param n n=0 means automagic */
		void set_normalization_const(float64_t n=0);

		/** get normalization constant */
		inline float64_t get_normalization_const()
		{
			return normalization_const;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "WDFeatures"; }

	protected:
		/** create wd kernel weighting heuristic */
		void set_wd_weights();

	protected:
		/** stringfeatures the wdfeatures are based on*/
		CStringFeatures<uint8_t>* strings;

		/** degree */
		int32_t degree;
		/** from degree */
		int32_t from_degree;
		/** length of string in vector */
		int32_t string_length;
		/** number of strings */
		int32_t num_strings;
		/** size of alphabet */
		int32_t alphabet_size;
		/** w dim */
		int32_t w_dim;
		/** wd weights */
		float64_t* wd_weights;

		/** normalization const */
		float64_t normalization_const;

};
}
#endif // _WDFEATURES_H___
