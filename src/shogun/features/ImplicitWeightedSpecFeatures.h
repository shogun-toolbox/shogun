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

#ifndef _IMPLICITSPECFEATURES_H___
#define _IMPLICITSPECFEATURES_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{

template <class ST> class CStringFeatures;

/** @brief Features that compute the Weighted Spectrum Kernel feature space
 * explicitly.
 *
 * \sa CWeightedCommWordStringKernel
 */
class CImplicitWeightedSpecFeatures : public CDotFeatures
{
	public:
		/** default constructor  */
		CImplicitWeightedSpecFeatures();

		/** constructor
		 *
		 * @param str stringfeatures (of words)
		 * @param normalize whether to use sqrtdiag normalization
		 */
		CImplicitWeightedSpecFeatures(CStringFeatures<uint16_t>* str, bool normalize=true);

		/** copy constructor */
		CImplicitWeightedSpecFeatures(const CImplicitWeightedSpecFeatures & orig);

		virtual ~CImplicitWeightedSpecFeatures();

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

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

		/** get number of strings
		 *
		 * @return number of strings
		 */
		virtual int32_t get_num_vectors() const;

		/** set weighted degree weights
		 *
		 * @return if setting was successful
		 */
		bool set_wd_weights();

		/** set custom weights (swig compatible)
		 *
		 * @param w weights
		 * @param d degree (must match number of weights)
		 * @return if setting was successful
		 */
		bool set_weights(float64_t* w, int32_t d);

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for weighted spectrum features */
		struct wspec_feature_iterator
		{
			/** pointer to feature vector */
			uint16_t* vec;
			/** index of vector */
			int32_t vidx;
			/** length of vector */
			int32_t vlen;
			/** if we need to free the vector*/
			bool vfree;

			/** @name Internal Parameters
			 * parameters of interal feature gen loop
			 */
			//@{
			int32_t offs;
			int32_t d;
			int32_t j;
			uint8_t mask;
			float64_t alpha;
			//@}
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
		virtual const char* get_name() const { return "ImplicitWeightedSpecFeatures"; }

	protected:
		/** compute the sqrt diag normalization constant per string
		 *
		 * \sa CSqrtDiagKernelNormalization
		 */
		void compute_normalization_const();

	protected:
		/** reference to strings */
		CStringFeatures<uint16_t>* strings;

		/** use sqrtdiag normalization */
		float64_t* normalization_factors;
		/** number of strings */
		int32_t num_strings;
		/** size of alphabet */
		int32_t alphabet_size;

		/** degree */
		int32_t degree;
		/** size of k-mer spectrum*/
		int32_t spec_size;

		/** weights for each of the subkernels of degree 1...d */
		float64_t* spec_weights;
};
}
#endif // _IMPLICITSPECFEATURES_H___
