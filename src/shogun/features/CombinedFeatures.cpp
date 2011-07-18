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
#include <shogun/io/io.h>

using namespace shogun;

void
CCombinedFeatures::init(void)
{
	m_parameters->add(&num_vec, "num_vec",
					  "Number of vectors.");
	m_parameters->add((CSGObject**) &feature_list,
					  "feature_list", "Feature list.");
}

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
