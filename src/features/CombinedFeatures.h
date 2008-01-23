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
class CCombinedFeatures;

class CCombinedFeatures : public CFeatures
{
public:
	CCombinedFeatures();
	CCombinedFeatures(const CCombinedFeatures& orig);
	virtual CFeatures* duplicate() const;
	virtual ~CCombinedFeatures();

	inline virtual EFeatureType get_feature_type()
	{
		return F_UNKNOWN;
	}
		
	inline virtual EFeatureClass get_feature_class()
	{
		return C_COMBINED;
	}

	inline virtual INT get_num_vectors()
	{
		if (feature_list->get_current_element())
		{
			return feature_list->get_current_element()->get_num_vectors();
		}
		else 
			return 0;
	}

	virtual INT get_size()
	{
		if (feature_list->get_current_element())
		{
			return feature_list->get_current_element()->get_size();
		}
		else 
			return 0;
	}

	void list_feature_objs();
	bool check_feature_obj_compatibility(CCombinedFeatures* comb_feat);

	inline CFeatures* get_first_feature_obj()
	{
        CFeatures* f=feature_list->get_first_element();
        SG_REF(f);
		return f;
	}
	inline CFeatures* get_first_feature_obj(CListElement<CFeatures*>*&current)
	{
		CFeatures* f=feature_list->get_first_element(current);
        SG_REF(f);
		return f;
	}

	inline CFeatures* get_next_feature_obj()
	{
		CFeatures* f=feature_list->get_next_element();
        SG_REF(f);
		return f;
	}
	inline CFeatures* get_next_feature_obj(CListElement<CFeatures*>*&current)
	{
		CFeatures* f=feature_list->get_next_element(current);
        SG_REF(f);
		return f;
	}

	inline CFeatures* get_last_feature_obj()
	{
		CFeatures* f=feature_list->get_last_element();
        SG_REF(f);
		return f;
	}

	inline bool insert_feature_obj(CFeatures* obj)
	{
        ASSERT(obj);
        SG_REF(obj);
		return feature_list->insert_element(obj);
	}

	inline bool append_feature_obj(CFeatures* obj)
	{
        ASSERT(obj);
        SG_REF(obj);
		return feature_list->append_element(obj);
	}

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

	inline int get_num_feature_obj()
	{
		return feature_list->get_num_elements();
	}
	
protected:
	CList<CFeatures*>* feature_list;
};
#endif
