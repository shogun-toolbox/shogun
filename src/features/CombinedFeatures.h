/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CCOMBINEDFEATURES__H__
#define _CCOMBINEDFEATURES__H__

#include "features/Features.h"
#include "lib/List.h"

class CFeatures;

/** The class CombinedFeatures is used to combine a number of of feature objects
 * into a single CombinedFeatures object. It keeps pointers to the added
 * sub-features.
 *
 * It is especially useful to combine kernels working on different domains
 * (c.f. CCombinedKernel) and to combine kernels looking at independent
 * features.
 */
class CCombinedFeatures : public CFeatures
{
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
			if (feature_list->get_current_element())
			{
				return feature_list->get_current_element()->get_num_vectors();
			}
			else 
				return 0;
		}

		/** get memory footprint of one feature
		 *
		 * @return memory footprint of one feature
		 */
		virtual int32_t get_size()
		{
			if (feature_list->get_current_element())
			{
				return feature_list->get_current_element()->get_size();
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
			CFeatures* f=feature_list->get_first_element();
			SG_REF(f);
			return f;
		}

		/** get first feature object
		 *
		 * @param current list of features
		 * @return first feature object
		 */
		inline CFeatures* get_first_feature_obj(CListElement<CFeatures*>*&current)
		{
			CFeatures* f=feature_list->get_first_element(current);
			SG_REF(f);
			return f;
		}

		/** get next feature object
		 *
		 * @return next feature object
		 */
		inline CFeatures* get_next_feature_obj()
		{
			CFeatures* f=feature_list->get_next_element();
			SG_REF(f);
			return f;
		}

		/** get next feature object
		 *
		 * @param current list of features
		 * @return next feature object
		 */
		inline CFeatures* get_next_feature_obj(CListElement<CFeatures*>*&current)
		{
			CFeatures* f=feature_list->get_next_element(current);
			SG_REF(f);
			return f;
		}

		/** get last feature object
		 *
		 * @return last feature object
		 */
		inline CFeatures* get_last_feature_obj()
		{
			CFeatures* f=feature_list->get_last_element();
			SG_REF(f);
			return f;
		}

		/** insert feature object
		 *
		 * @param obj feature object to insert
		 * @return if inserting was successful
		 */
		inline bool insert_feature_obj(CFeatures* obj)
		{
			ASSERT(obj);
			SG_REF(obj);
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
			SG_REF(obj);
			return feature_list->append_element(obj);
		}

		/** delete feature object
		 *
		 * @return if deleting was succesful
		 */
		inline bool delete_feature_obj()
		{
			CFeatures* f=feature_list->delete_element();
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

	protected:
		/** feature list */
		CList<CFeatures*>* feature_list;
};
#endif
