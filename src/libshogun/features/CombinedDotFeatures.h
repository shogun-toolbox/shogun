/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _COMBINEDDOTFEATURES_H___
#define _COMBINEDDOTFEATURES_H___

#include "lib/common.h"
#include "lib/List.h"
#include "features/DotFeatures.h"
#include "features/Features.h"

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

		inline virtual int32_t get_num_vectors()
		{
			return num_vectors;
		}

		/** obtain the dimensionality of the feature space
		 *
		 * @return dimensionality
		 */
		inline virtual int32_t get_dim_feature_space()
		{
			return  num_dimensions;
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
		virtual int32_t get_nnz_features_for_vector(int32_t num);

		/** get feature type
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type()
		{
			return F_DREAL;
		}

		/** get feature class
		 *
		 * @return feature class
		 */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_COMBINED_DOT;
		}

		inline virtual int32_t get_size()
		{
			return sizeof(float64_t);
		}

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** list feature objects */
		void list_feature_objs();

		/** get first feature object
		 *
		 * @return first feature object
		 */
		inline CDotFeatures* get_first_feature_obj()
		{
			CDotFeatures* f=feature_list->get_first_element();
			SG_REF(f);
			return f;
		}

		/** get first feature object
		 *
		 * @param current list of features
		 * @return first feature object
		 */
		inline CDotFeatures* get_first_feature_obj(CListElement<CDotFeatures*>*&current)
		{
			CDotFeatures* f=feature_list->get_first_element(current);
			SG_REF(f);
			return f;
		}

		/** get next feature object
		 *
		 * @return next feature object
		 */
		inline CDotFeatures* get_next_feature_obj()
		{
			CDotFeatures* f=feature_list->get_next_element();
			SG_REF(f);
			return f;
		}

		/** get next feature object
		 *
		 * @param current list of features
		 * @return next feature object
		 */
		inline CDotFeatures* get_next_feature_obj(CListElement<CDotFeatures*>*&current)
		{
			CDotFeatures* f=feature_list->get_next_element(current);
			SG_REF(f);
			return f;
		}

		/** get last feature object
		 *
		 * @return last feature object
		 */
		inline CDotFeatures* get_last_feature_obj()
		{
			CDotFeatures* f=feature_list->get_last_element();
			SG_REF(f);
			return f;
		}

		/** insert feature object
		 *
		 * @param obj feature object to insert
		 * @return if inserting was successful
		 */
		inline bool insert_feature_obj(CDotFeatures* obj)
		{
			ASSERT(obj);
			SG_REF(obj);
			bool result=feature_list->insert_element(obj);
			update_dim_feature_space_and_num_vec();
			return result;
		}

		/** append feature object
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		inline bool append_feature_obj(CDotFeatures* obj)
		{
			ASSERT(obj);
			SG_REF(obj);
			bool result=feature_list->append_element(obj);
			update_dim_feature_space_and_num_vec();
			return result;
		}

		/** delete feature object
		 *
		 * @return if deleting was succesful
		 */
		inline bool delete_feature_obj()
		{
			CDotFeatures* f=feature_list->delete_element();
			if (f)
			{
				SG_UNREF(f);
				update_dim_feature_space_and_num_vec();
				return true;
			}
			else
				return false;
		}

		/** get number of feature objects
		 *
		 * @return number of feature objects
		 */
		inline int32_t get_num_feature_obj()
		{
			return feature_list->get_num_elements();
		}

		/** get subfeature weights
		 *
		 * @param weights subfeature weights
		 * @param num_weights where number of weights is stored
		 */
		virtual void get_subfeature_weights(float64_t** weights, int32_t* num_weights);

		/** set subfeature weights
		 *
		 * @param weights new subfeature weights
		 * @param num_weights number of subfeature weights
		 */
		virtual void set_subfeature_weights(
			float64_t* weights, int32_t num_weights);

		/** @return object name */
		inline virtual const char* get_name() const { return "CombinedDotFeatures"; }

	protected:
		/** update total number of dimensions and vectors */
		void update_dim_feature_space_and_num_vec();

	protected:
		/** feature list */
		CList<CDotFeatures*>* feature_list;

		/// total number of vectors
		int32_t num_vectors;
		/// total number of dimensions
		int32_t num_dimensions;
};
#endif // _DOTFEATURES_H___
