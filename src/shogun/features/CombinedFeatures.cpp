/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/CombinedFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Set.h>
#include <shogun/lib/Map.h>

using namespace shogun;

CCombinedFeatures::CCombinedFeatures()
: CFeatures(0)
{
	init();

	feature_array = new CDynamicObjectArray();
	num_vec=0;
}

CCombinedFeatures::CCombinedFeatures(const CCombinedFeatures& orig)
: CFeatures(0)
{
	init();

	feature_array=new CDynamicObjectArray();
	//TODO copy features
	num_vec=orig.num_vec;
}

CFeatures* CCombinedFeatures::duplicate() const
{
	return new CCombinedFeatures(*this);
}

CCombinedFeatures::~CCombinedFeatures()
{
	SG_UNREF(feature_array);
}

CFeatures* CCombinedFeatures::get_feature_obj(int32_t idx)
{
	return (CFeatures*) feature_array->get_element(idx);
} 

void CCombinedFeatures::list_feature_objs()
{
	SG_INFO("BEGIN COMBINED FEATURES LIST - ")
	this->list_feature_obj();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* f = get_feature_obj(f_idx);
		f->list_feature_obj();
		SG_UNREF(f);
	}

	SG_INFO("END COMBINED FEATURES LIST - ")
}

bool CCombinedFeatures::check_feature_obj_compatibility(CCombinedFeatures* comb_feat)
{
	bool result=false;

	if ( (comb_feat) && (this->get_num_feature_obj() == comb_feat->get_num_feature_obj()) )
	{
		for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
		{
			CFeatures* f1=this->get_feature_obj(f_idx);
			CFeatures* f2=comb_feat->get_feature_obj(f_idx);

			if ( ! (f1 && f2 && f1->check_feature_compatibility(f2)) )
			{
				SG_UNREF(f1);
				SG_UNREF(f2);
				SG_INFO("not compatible, combfeat\n")
				comb_feat->list_feature_objs();
				SG_INFO("vs this\n")
				this->list_feature_objs();
				return false;
			}
			
			SG_UNREF(f1);
			SG_UNREF(f2);
		}
		SG_DEBUG("features are compatible\n")
		result=true;
	}
	else
	{
		if (!comb_feat)
		{
			SG_WARNING("comb_feat is NULL \n");
		}
		else
		{
			SG_WARNING("number of features in combined feature objects differs (%d != %d)\n", this->get_num_feature_obj(), comb_feat->get_num_feature_obj())
				SG_INFO("compare\n")
				comb_feat->list_feature_objs();
			SG_INFO("vs this\n")
				this->list_feature_objs();
		}
	}

	return result;
}

CFeatures* CCombinedFeatures::get_first_feature_obj()
{
	return get_feature_obj(0);
}

CFeatures* CCombinedFeatures::get_last_feature_obj()
{
	return get_feature_obj(get_num_feature_obj()-1);
}

bool CCombinedFeatures::insert_feature_obj(CFeatures* obj, int32_t idx)
{
	ASSERT(obj)
	int32_t n=obj->get_num_vectors();

	if (get_num_vectors()>0 && n!=get_num_vectors())
	{
		SG_ERROR("Number of feature vectors does not match (expected %d, "
				"obj has %d)\n", get_num_vectors(), n);
	}

	num_vec=n;
	return feature_array->insert_element(obj, idx);
}

bool CCombinedFeatures::append_feature_obj(CFeatures* obj)
{
	ASSERT(obj)
	int32_t n=obj->get_num_vectors();

	if (get_num_vectors()>0 && n!=get_num_vectors())
	{
		SG_ERROR("Number of feature vectors does not match (expected %d, "
				"obj has %d)\n", get_num_vectors(), n);
	}

	num_vec=n;

	int num_feature_obj = get_num_feature_obj();
	feature_array->push_back(obj);
	return num_feature_obj+1 == feature_array->get_num_elements();
}

bool CCombinedFeatures::delete_feature_obj(int32_t idx)
{
	return feature_array->delete_element(idx);
}

int32_t CCombinedFeatures::get_num_feature_obj()
{
	return feature_array->get_num_elements();
}

void CCombinedFeatures::init()
{
	m_parameters->add(&num_vec, "num_vec",
					  "Number of vectors.");
	m_parameters->add((CSGObject**) &feature_array,
					  "feature_array", "Feature array.");
}

CFeatures* CCombinedFeatures::create_merged_copy(CFeatures* other)
{
	/* TODO, if all features are the same, only one copy should be created
	 * in memory */
	SG_WARNING("Heiko Strathmann: FIXME, unefficient!\n")

	SG_DEBUG("entering %s::create_merged_copy()\n", get_name())
	if (get_feature_type()!=other->get_feature_type() ||
			get_feature_class()!=other->get_feature_class() ||
			strcmp(get_name(), other->get_name()))
	{
		SG_ERROR("%s::create_merged_copy(): Features are of different type!\n",
				get_name());
	}

	CCombinedFeatures* casted=dynamic_cast<CCombinedFeatures*>(other);

	if (!casted)
	{
		SG_ERROR("%s::create_merged_copy(): Could not cast object of %s to "
				"same type as %s\n",get_name(), other->get_name(), get_name());
	}

	if (get_num_feature_obj()!=casted->get_num_feature_obj())
	{
		SG_ERROR("%s::create_merged_copy(): Only possible if both instances "
				"have the same number of sub-feature-objects\n", get_name());
	}

	CCombinedFeatures* result=new CCombinedFeatures();
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current_this=get_feature_obj(f_idx);
		CFeatures* current_other=casted->get_feature_obj(f_idx);

		result->append_feature_obj(
				current_this->create_merged_copy(current_other));
		SG_UNREF(current_this);
		SG_UNREF(current_other);
	}

	SG_DEBUG("leaving %s::create_merged_copy()\n", get_name())
	return result;
}

void CCombinedFeatures::add_subset(SGVector<index_t> subset)
{
	SG_DEBUG("entering %s::add_subset()\n", get_name())
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{	
		CFeatures* current=get_feature_obj(f_idx);
	
		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->add_subset(subset);
			processed->add(current);
			SG_DEBUG("adding subset to %s at %p\n",
					current->get_name(), current);
		}
		SG_UNREF(current);
	}

	/* also add subset to local stack to have it for easy access */
	m_subset_stack->add_subset(subset);

	subset_changed_post();
	SG_UNREF(processed);
	SG_DEBUG("leaving %s::add_subset()\n", get_name())
}

void CCombinedFeatures::remove_subset()
{
	SG_DEBUG("entering %s::remove_subset()\n", get_name())
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);
		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->remove_subset();
			processed->add(current);
			SG_DEBUG("removing subset from %s at %p\n",
					current->get_name(), current);
		}
		SG_UNREF(current);
	}

	/* also remove subset from local stack to have it for easy access */
	m_subset_stack->remove_subset();

	subset_changed_post();
	SG_UNREF(processed);
	SG_DEBUG("leaving %s::remove_subset()\n", get_name())
}

void CCombinedFeatures::remove_all_subsets()
{
	SG_DEBUG("entering %s::remove_all_subsets()\n", get_name())
	CSet<CFeatures*>* processed=new CSet<CFeatures*>();

	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);
		if (!processed->contains(current))
		{
			/* remember that subset was added here */
			current->remove_all_subsets();
			processed->add(current);
			SG_DEBUG("removing all subsets from %s at %p\n",
					current->get_name(), current);
		}
		SG_UNREF(current);
	}

	/* also remove subsets from local stack to have it for easy access */
	m_subset_stack->remove_all_subsets();

	subset_changed_post();
	SG_UNREF(processed);
	SG_DEBUG("leaving %s::remove_all_subsets()\n", get_name())
}

CFeatures* CCombinedFeatures::copy_subset(SGVector<index_t> indices)
{
	/* this is returned with the results of copy_subset of sub-features */
	CCombinedFeatures* result=new CCombinedFeatures();

	/* map to only copy same feature objects once */
	CMap<CFeatures*, CFeatures*>* processed=new CMap<CFeatures*, CFeatures*>();
	for (index_t f_idx=0; f_idx<get_num_feature_obj(); f_idx++)
	{
		CFeatures* current=get_feature_obj(f_idx);
	
		CFeatures* new_element=NULL;

		/* only copy if not done yet, otherwise, use old copy */
		if (!processed->contains(current))
		{
			new_element=current->copy_subset(indices);
			processed->add(current, new_element);
		}
		else
		{
			new_element=processed->get_element(current);

			/* has to be SG_REF'ed since it will be unrefed afterwards */
			SG_REF(new_element);
		}

		/* add to result */
		result->append_feature_obj(new_element);

		/* clean up: copy_subset of SG_REF has to be undone */
		SG_UNREF(new_element);

		SG_UNREF(current);
	}

	SG_UNREF(processed);

	SG_REF(result);
	return result;
}
