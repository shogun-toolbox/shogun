/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _HASHEDWDFEATURESTRANSPOSED_H___
#define _HASHEDWDFEATURESTRANSPOSED_H___

#include <lib/common.h>
#include <features/DotFeatures.h>
#include <features/StringFeatures.h>
#include <lib/Hash.h>

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief Features that compute the Weighted Degreee Kernel feature space
 * explicitly.
 *
 * \sa CWeightedDegreeStringKernel
 */
class CHashedWDFeaturesTransposed : public CDotFeatures
{
	public:
		/** default constructor  */
		CHashedWDFeaturesTransposed();

		/** constructor
		 *
		 * @param str stringfeatures (of bytes)
		 * @param start_order do degrees starting with start_order up to order
		 * @param order of wd kernel
		 * @param from_order use first order weights from higher order weighting
		 * @param hash_bits number of bits in hash
		 */
		CHashedWDFeaturesTransposed(CStringFeatures<uint8_t>* str, int32_t start_order,
				int32_t order,  int32_t from_order, int32_t hash_bits=12);

		/** copy constructor */
		CHashedWDFeaturesTransposed(const CHashedWDFeaturesTransposed & orig);

		/** destructor */
		virtual ~CHashedWDFeaturesTransposed();

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const
		{
			return w_dim;
		}

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
		virtual void dense_dot_range_subset(int32_t* sub_index, int32_t num, float64_t* output, float64_t* alphas, float64_t* vec, int32_t dim, float64_t b);


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
		virtual int32_t get_nnz_features_for_vector(int32_t num)
		{
			return w_dim/alphabet_size;
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
		virtual EFeatureType get_feature_type() const
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const
		{
			return C_WD;
		}

		virtual int32_t get_num_vectors() const
		{
			return num_strings;
		}

		/** set normalization constant
		 * @param n n=0 means automagic */
		void set_normalization_const(float64_t n=0);

		/** get normalization constant */
		inline float64_t get_normalization_const()
		{
			return normalization_const;
		}

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct hashed_wd_transposed_feature_iterator
		{
			/** pointer to feature vector */
			uint16_t* vec;
			/** index of vector */
			int32_t vidx;
			/** length of vector */
			int32_t vlen;
			/** if we need to free the vector*/
			bool vfree;

			/** feature index */
			int32_t index;

		};
		#endif

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

		/** @return object name */
		virtual const char* get_name() const { return "HashedWDFeaturesTransposed"; }

	protected:

		/** create wd kernel weighting heuristic */
		void set_wd_weights();

		/** helper function for parallel dense_dot computation */
		static void* dense_dot_range_helper(void* p);

	protected:
		/** stringfeatures the wdfeatures are based on*/
		CStringFeatures<uint8_t>* strings;

		/** pointer to transposed strings */
		SGString<uint8_t>* transposed_strings;

		/** degree */
		int32_t degree;
		/** start_degree */
		int32_t start_degree;
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
		/** partial w dim == hashsize*/
		int32_t partial_w_dim;
		/** wd weights */
		float64_t* wd_weights;
		/** mask */
		uint32_t mask;
		/** number of bits in hash */
		int32_t m_hash_bits;

		/** normalization const */
		float64_t normalization_const;
};
}
#endif // _HASHEDWDFEATURESTRANSPOSED_H___
