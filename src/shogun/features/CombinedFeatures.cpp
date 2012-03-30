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

#include <shogun/features/CombinedFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CCombinedFeatures::CCombinedFeatures()
: CFeatures(0)
{
	init();

	feature_list=new CList(true);
	num_vec=0;
}

CCombinedFeatures::CCombinedFeatures(const CCombinedFeatures & orig)
: CFeatures(0)
{
	init();

	feature_list=new CList(true);
	//todo copy features
	num_vec=orig.num_vec;
}

CFeatures* CCombinedFeatures::duplicate() const
{
	return new CCombinedFeatures(*this);
}

CCombinedFeatures::~CCombinedFeatures()
{
	SG_UNREF(feature_list);
}

int32_t CCombinedFeatures::get_size()
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

void CCombinedFeatures::list_feature_objs()
{
	SG_INFO( "BEGIN COMBINED FEATURES LIST - ");
	this->list_feature_obj();

	CListElement* current = NULL ;
	CFeatures* f=get_first_feature_obj(current);

	while (f)
	{
		f->list_feature_obj();
		SG_UNREF(f);
		f=get_next_feature_obj(current);
	}

	SG_INFO( "END COMBINED FEATURES LIST - ");
}

bool CCombinedFeatures::check_feature_obj_compatibility(CCombinedFeatures* comb_feat)
{
	bool result=false;

	if (comb_feat && (this->get_num_feature_obj() == comb_feat->get_num_feature_obj()) )
	{
		CFeatures* f1=this->get_first_feature_obj();
		CFeatures* f2=comb_feat->get_first_feature_obj();

		if (f1 && f2 && f1->check_feature_compatibility(f2))
		{
			SG_UNREF(f1);
			SG_UNREF(f2);
			while( ( (f1=this->get_next_feature_obj()) != NULL )  &&
				   ( (f2=comb_feat->get_next_feature_obj()) != NULL) )
			{
				if (!f1->check_feature_compatibility(f2))
				{
					SG_UNREF(f1);
					SG_UNREF(f2);
					SG_INFO( "not compatible, combfeat\n");
					comb_feat->list_feature_objs();
					SG_INFO( "vs this\n");
					this->list_feature_objs();
					return false;
				}
				SG_UNREF(f1);
				SG_UNREF(f2);
			}

			SG_DEBUG( "features are compatible\n");
			result=true;
		}
		else
			SG_WARNING( "first 2 features not compatible\n");
	}
	else
	{
		SG_WARNING( "number of features in combined feature objects differs (%d != %d)\n", this->get_num_feature_obj(), comb_feat->get_num_feature_obj());
		SG_INFO( "compare\n");
		comb_feat->list_feature_objs();
		SG_INFO( "vs this\n");
		this->list_feature_objs();
	}

	return result;
}

CFeatures* CCombinedFeatures::get_first_feature_obj()
{
	return (CFeatures*) feature_list->get_first_element();
}

CFeatures* CCombinedFeatures::get_first_feature_obj(CListElement*& current)
{
	return (CFeatures*) feature_list->get_first_element(current);
}

CFeatures* CCombinedFeatures::get_next_feature_obj()
{
	return (CFeatures*) feature_list->get_next_element();
}

CFeatures* CCombinedFeatures::get_next_feature_obj(CListElement*& current)
{
	return (CFeatures*) feature_list->get_next_element(current);
}

CFeatures* CCombinedFeatures::get_last_feature_obj()
{
	return (CFeatures*) feature_list->get_last_element();
}

bool CCombinedFeatures::insert_feature_obj(CFeatures* obj)
{
	ASSERT(obj);
	int32_t n=obj->get_num_vectors();

	if (num_vec>0 && n!=num_vec)
		SG_ERROR("Number of feature vectors does not match (expected %d, obj has %d)\n", num_vec, n);

	num_vec=n;
	return feature_list->insert_element(obj);
}

bool CCombinedFeatures::append_feature_obj(CFeatures* obj)
{
	ASSERT(obj);
	int32_t n=obj->get_num_vectors();

	if (num_vec>0 && n!=num_vec)
		SG_ERROR("Number of feature vectors does not match (expected %d, obj has %d)\n", num_vec, n);

	num_vec=n;
	return feature_list->append_element(obj);
}

bool CCombinedFeatures::delete_feature_obj()
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

int32_t CCombinedFeatures::get_num_feature_obj()
{
	return feature_list->get_num_elements();
}

void CCombinedFeatures::init()
{
	m_parameters->add(&num_vec, "num_vec",
					  "Number of vectors.");
	m_parameters->add((CSGObject**) &feature_list,
					  "feature_list", "Feature list.");
}
