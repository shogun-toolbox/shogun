/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/CombinedFeatures.h"
#include "lib/io.h"

class CCombinedFeatures;

CCombinedFeatures::CCombinedFeatures() : CFeatures(0)
{
	feature_list=new CList<CFeatures*>(true);
}

CCombinedFeatures::CCombinedFeatures(const CCombinedFeatures & orig) : CFeatures(0)
{
}

CFeatures* CCombinedFeatures::duplicate() const
{
	return new CCombinedFeatures(*this);
}

CCombinedFeatures::~CCombinedFeatures()
{
	delete feature_list;
}

void CCombinedFeatures::list_feature_objs()
{
	CFeatures* f;

	CIO::message(M_INFO, "BEGIN COMBINED FEATURES LIST - ");
	this->list_feature_obj();

	CListElement<CFeatures*> * current = NULL ;
	f=get_first_feature_obj(current);

	while (f)
	{
		f->list_feature_obj();
		f=get_next_feature_obj(current);
	}

	CIO::message(M_INFO, "END COMBINED FEATURES LIST - ");
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
			while( ( (f1=this->get_next_feature_obj()) != NULL )  && 
				   ( (f2=comb_feat->get_next_feature_obj()) != NULL) )
			{
				if (!f1->check_feature_compatibility(f2))
				{
					CIO::message(M_INFO, "not compatible, combfeat\n");
					comb_feat->list_feature_objs();
					CIO::message(M_INFO, "vs this\n");
					this->list_feature_objs();
					return false;
				}
			}

			CIO::message(M_DEBUG, "features are compatible\n");
			result=true;
		}
		else
			CIO::message(M_WARN, "first 2 features not compatible\n");
	}
	else
	{
		CIO::message(M_WARN, "number of features in combined feature objects differs (%d != %d)\n", this->get_num_feature_obj(), comb_feat->get_num_feature_obj());
		CIO::message(M_INFO, "compare\n");
		comb_feat->list_feature_objs();
		CIO::message(M_INFO, "vs this\n");
		this->list_feature_objs();
	}

	return result;
}
