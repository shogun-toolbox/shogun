/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _SNPFEATURES_H___
#define _SNPFEATURES_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
template <class ST> class CStringFeatures;

/** @brief Features that compute the Weighted Degreee Kernel feature space
 * explicitly.
 *
 * \sa CWeightedDegreeStringKernel
 */
class CSNPFeatures : public CDotFeatures
{
	public:
		/** default constructor  */
		CSNPFeatures();

		/** constructor
		 *
		 * @param str stringfeatures (of bytes)
		 */
		CSNPFeatures(CStringFeatures<uint8_t>* str);

		/** copy constructor */
		CSNPFeatures(const CSNPFeatures & orig);

		/** destructor */
		virtual ~CSNPFeatures();

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
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
				float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
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

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** get feature type
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get number of vectors
		 *
		 * @return number of vectors
		 */
		virtual int32_t get_num_vectors() const;

		/** set normalization constant
		 * @param n n=0 means automagic */
		void set_normalization_const(float64_t n=0);

		/** get normalization constant */
		float64_t get_normalization_const();

		/** set the minor base string
		 *
		 * @param str base string
		 */
		void set_minor_base_string(const char* str);

		/** set the major base string
		 *
		 * @param str base string
		 */
		void set_major_base_string(const char* str);

		/** get the minor base string
		 *
		 * @return the minor base string
		 */
		char* get_minor_base_string();

		/** return the major base string
		 *
		 * @return major base string
		 */
		char* get_major_base_string();

		/** compute the base strings from current strings optionally taking
		 * into account snp
		 *
		 * @param snp optionally compute base string for snp too
		 */
		void obtain_base_strings(CSNPFeatures* snp=NULL);

		/** @return object name */
		virtual const char* get_name() const { return "SNPFeatures"; }

		/** compute histogram over strings
		 */
		virtual SGMatrix<float64_t> get_histogram(bool normalize=true);

		/**
		 * compute 2x3 histogram table
		 */
		static SGMatrix<float64_t> get_2x3_table(CSNPFeatures* pos, CSNPFeatures* neg);

	private:
		/** determine minor and major base strings from current strings
		 * @arg minor - array of string_length inited with zero that will
		 *              contain the minor base string
		 * @arg major - array of string_length inited with zero that will
		 *              contain the major base string
		 */
		void find_minor_major_strings(uint8_t* minor, uint8_t* major);

	protected:
		/** stringfeatures the wdfeatures are based on*/
		CStringFeatures<uint8_t>* strings;

		/** length of string in vector */
		int32_t string_length;
		/** number of strings */
		int32_t num_strings;
		/** dim of feature space */
		int32_t w_dim;

		/** normalization const */
		float64_t normalization_const;

		/** allele A */
		uint8_t* m_str_min;
		/** allele B */
		uint8_t* m_str_maj;
};
}
#endif // _SNPFEATURES_H___
