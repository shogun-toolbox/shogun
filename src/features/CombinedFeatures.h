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

	inline CFeatures* get_first_feature_obj()
	{
		return feature_list->get_first_element();
	}

	inline CFeatures* get_next_feature_obj()
	{
		return feature_list->get_next_element();
	}

	inline bool insert_feature_obj(CFeatures* obj)
	{
		return feature_list->insert_element(obj);
	}

	inline bool append_feature_obj(CFeatures* obj)
	{
		return feature_list->append_element(obj);
	}

	inline bool delete_feature_obj()
	{
		return feature_list->delete_element();
	}

	inline int get_num_feature_obj()
	{
		return feature_list->get_num_elements();
	}
	
protected:
	CList<CFeatures*>* feature_list;
};
#endif
