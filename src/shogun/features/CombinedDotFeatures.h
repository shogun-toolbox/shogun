/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009-2010 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 *
 */

#ifndef _COMBINEDDOTFEATURES_H___
#define _COMBINEDDOTFEATURES_H___

#include <lib/common.h>
#include <lib/DynamicObjectArray.h>
#include <features/DotFeatures.h>

namespace shogun
{
class CFeatures;
class CDynamicObjectArray;
/** @brief Features that allow stacking of a number of DotFeatures.
 *
 * They transparently provide all the operations of DotFeatures, i.e.
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
 *   \f[{\bf z'} = \alpha {\bf x} + {\bf z}\f]
 *
 */
class CCombinedDotFeatures : public CDotFeatures
{
	public:
		/** constructor */
		CCombinedDotFeatures();

		/** copy constructor */
		CCombinedDotFeatures(const CCombinedDotFeatures & orig);

		/** destructor */
		virtual ~CCombinedDotFeatures();

		/** get the number of vectors
		 *
		 * @return number of vectors
		 */
		virtual int32_t get_num_vectors() const
		{
			return num_vectors;
		}

		/** obtain the dimensionality of the feature space
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const
		{
			return  num_dimensions;
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
		virtual void dense_dot_range(float64_t* output, int32_t start,
				int32_t stop, float64_t* alphas, float64_t* vec,
				int32_t dim, float64_t b);

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
		virtual void dense_dot_range_subset(int32_t* sub_index, int32_t num,
				float64_t* output, float64_t* alphas, float64_t* vec,
				int32_t dim, float64_t b);

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
		virtual EFeatureType get_feature_type() const
		{
			return F_DREAL;
		}

		/** get feature class
		 *
		 * @return feature class
		 */
		virtual EFeatureClass get_feature_class() const
		{
			return C_COMBINED_DOT;
		}

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for combined dotfeatures */
		struct combined_feature_iterator
		{
			/** pointer to current feature object */
			CDotFeatures* f;
			/** pointer to combined feature iterator */
			void* iterator;
			/** the index of the vector over whose components to iterate over */
			int32_t vector_index;
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

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** list feature objects */
		void list_feature_objs();

		/** get feature object at position idx
		 *
		 * @param idx the index of the feature to return
		 * @return next feature object
		 */
		CDotFeatures* get_feature_obj(int32_t idx);

		/** insert feature object at position idx
		 *  idx must be < get_num_feature_obj()
		 *
		 * @param obj feature object to insert
		 * @param idx position where to insert the feature obj
		 * @return if inserting was successful
		 */
		bool insert_feature_obj(CDotFeatures* obj, int32_t idx);

		/** append feature object
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		bool append_feature_obj(CDotFeatures* obj);

		/** delete feature object at position idx
		 *
		 * @param idx the index of the feature object to delete
		 * @return if deleting was successful
		 */
		bool delete_feature_obj(int32_t idx);

		/** get number of feature objects
		 *
		 * @return number of feature objects
		 */
		int32_t get_num_feature_obj();

		/** get subfeature weights
		 *
		 */
		virtual SGVector<float64_t> get_subfeature_weights();

		/** set subfeature weights
		 *
		 * @param weights new subfeature weights
		 */
		virtual void set_subfeature_weights(SGVector<float64_t> weights);

		/** @return object name */
		virtual const char* get_name() const { return "CombinedDotFeatures"; }

	protected:
		/** update total number of dimensions and vectors */
		void update_dim_feature_space_and_num_vec();

	private:
		void init();

	protected:
		/** feature array */
		CDynamicObjectArray* feature_array;

		/// idx for iterator
		int32_t iterator_idx;
		/// total number of vectors
		int32_t num_vectors;
		/// total number of dimensions
		int32_t num_dimensions;
};
}
#endif // _DOTFEATURES_H___
