/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include "features/Features.h"
#include "lib/List.h"

namespace shogun
{
class CFeatures;
class CList;
class CListElement;

/** @brief The class CombinedFeatures is used to combine a number of of feature objects
 * into a single CombinedFeatures object.
 *
 * It keeps pointers to the added sub-features and is especially useful to
 * combine kernels working on different domains (c.f. CCombinedKernel) and to
 * combine kernels looking at independent features.
 */
class CCombinedFeatures : public CFeatures
{
	void init(void);

	public:
		/** default constructor */
		CCombinedFeatures();
		/** copy constructor */
		CCombinedFeatures(const CCombinedFeatures& orig);

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** destructor */
		virtual ~CCombinedFeatures();

		/** get feature type
		 *
		 * @return feature type UNKNOWN
		 */
		inline virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class SIMPLE
		 */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_COMBINED;
		}

		/** get number of feature vectors
		 *
		 * @return number of feature vectors
		 */
		inline virtual int32_t get_num_vectors()
		{
			return num_vec;
		}

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size()
		{
			CFeatures* f=(CFeatures*) feature_list
				->get_current_element();
			if (f)
			{
				int32_t s=f->get_size();
				SG_UNREF(f)
				return s;
			}
			else 
				return 0;
		}

		/** list feature objects */
		void list_feature_objs();

		/** check feature object compatibility
		 *
		 * @param comb_feat feature to check for compatibility
		 * @return if feature is compatible
		 */
		bool check_feature_obj_compatibility(CCombinedFeatures* comb_feat);

		/** get first feature object
		 *
		 * @return first feature object
		 */
		inline CFeatures* get_first_feature_obj()
		{
			return (CFeatures*) feature_list->get_first_element();
		}

		/** get first feature object
		 *
		 * @param current list of features
		 * @return first feature object
		 */
		inline CFeatures* get_first_feature_obj(CListElement*& current)
		{
			return (CFeatures*) feature_list->get_first_element(current);
		}

		/** get next feature object
		 *
		 * @return next feature object
		 */
		inline CFeatures* get_next_feature_obj()
		{
			return (CFeatures*) feature_list->get_next_element();
		}

		/** get next feature object
		 *
		 * @param current list of features
		 * @return next feature object
		 */
		inline CFeatures* get_next_feature_obj(CListElement*& current)
		{
			return (CFeatures*) feature_list->get_next_element(current);
		}

		/** get last feature object
		 *
		 * @return last feature object
		 */
		inline CFeatures* get_last_feature_obj()
		{
			return (CFeatures*) feature_list->get_last_element();
		}

		/** insert feature object
		 *
		 * @param obj feature object to insert
		 * @return if inserting was successful
		 */
		inline bool insert_feature_obj(CFeatures* obj)
		{
			ASSERT(obj);
			int32_t n=obj->get_num_vectors();

			if (num_vec>0 && n!=num_vec)
				SG_ERROR("Number of feature vectors does not match (expected %d, obj has %d)\n", num_vec, n);

			num_vec=n;
			return feature_list->insert_element(obj);
		}

		/** append feature object
		 *
		 * @param obj feature object to append
		 * @return if appending was successful
		 */
		inline bool append_feature_obj(CFeatures* obj)
		{
			ASSERT(obj);
			int32_t n=obj->get_num_vectors();

			if (num_vec>0 && n!=num_vec)
				SG_ERROR("Number of feature vectors does not match (expected %d, obj has %d)\n", num_vec, n);

			num_vec=n;
			return feature_list->append_element(obj);
		}

		/** delete feature object
		 *
		 * @return if deleting was successful
		 */
		inline bool delete_feature_obj()
		{
			CFeatures* f=(CFeatures*)feature_list->delete_element();
			if (f)
			{
				SG_UNREF(f);
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

		/** @return object name */
		inline virtual const char* get_name() const { return "CombinedFeatures"; }

	protected:
		/** feature list */
		CList* feature_list;

		/** number of vectors
		 * must match between sub features
		 */
		int32_t num_vec;
};
}
#endif
