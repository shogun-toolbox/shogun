/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 * 
 * Code adapted from CCombinedKernel
 */

#include <shogun/kernel/ProductKernel.h>
#include <shogun/kernel/CustomKernel.h>

using namespace shogun;

CProductKernel::CProductKernel(int32_t size)
: CKernel(size)
{
	init();

	SG_INFO("Product kernel created (%p)\n", this) ;
}

CProductKernel::~CProductKernel()
{
	cleanup();
	SG_UNREF(kernel_list);

	SG_INFO("Product kernel deleted (%p).\n", this);
}

//Adapted from CCombinedKernel
bool CProductKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l,r);
	ASSERT(l->get_feature_class()==C_COMBINED);
	ASSERT(r->get_feature_class()==C_COMBINED);
	ASSERT(l->get_feature_type()==F_UNKNOWN);
	ASSERT(r->get_feature_type()==F_UNKNOWN);

	CFeatures* lf=NULL;
	CFeatures* rf=NULL;
	CKernel* k=NULL;

	bool result=true;

	CListElement* lfc = NULL;
	CListElement* rfc = NULL;
	lf=((CCombinedFeatures*) l)->get_first_feature_obj(lfc);
	rf=((CCombinedFeatures*) r)->get_first_feature_obj(rfc);
	CListElement* current = NULL;
	k=get_first_kernel(current);

	while ( result && k )
	{
		// skip over features - the custom kernel does not need any
		if (k->get_kernel_type() != K_CUSTOM)
		{
			if (!lf || !rf)
			{
				SG_UNREF(lf);
				SG_UNREF(rf);
				SG_UNREF(k);
				SG_ERROR( "ProductKernel: Number of features/kernels does not match - bailing out\n");
			}

			SG_DEBUG( "Initializing 0x%p - \"%s\"\n", this, k->get_name());
			result=k->init(lf,rf);
			SG_UNREF(lf);
			SG_UNREF(rf);

			lf=((CCombinedFeatures*) l)->get_next_feature_obj(lfc) ;
			rf=((CCombinedFeatures*) r)->get_next_feature_obj(rfc) ;
		}
		else
		{
			SG_DEBUG( "Initializing 0x%p - \"%s\" (skipping init, this is a CUSTOM kernel)\n", this, k->get_name());
			if (!k->has_features())
				SG_ERROR("No kernel matrix was assigned to this Custom kernel\n");
			if (k->get_num_vec_lhs() != num_lhs)
				SG_ERROR("Number of lhs-feature vectors (%d) not match with number of rows (%d) of custom kernel\n", num_lhs, k->get_num_vec_lhs());
			if (k->get_num_vec_rhs() != num_rhs)
				SG_ERROR("Number of rhs-feature vectors (%d) not match with number of cols (%d) of custom kernel\n", num_rhs, k->get_num_vec_rhs());
		}

		SG_UNREF(k);
		k=get_next_kernel(current) ;
	}

	if (!result)
	{
		SG_INFO( "ProductKernel: Initialising the following kernel failed\n");
		if (k)
			k->list_kernel();
		else
			SG_INFO( "<NULL>\n");
		return false;
	}

	if ((lf!=NULL) || (rf!=NULL) || (k!=NULL))
	{
		SG_UNREF(lf);
		SG_UNREF(rf);
		SG_UNREF(k);
		SG_ERROR( "ProductKernel: Number of features/kernels does not match - bailing out\n");
	}

	initialized=true;
	return true;
}

//Adapted from CCombinedKernel
void CProductKernel::remove_lhs()
{
	CListElement* current = NULL ;
	CKernel* k=get_first_kernel(current);

	while (k)
	{
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs();

		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	CKernel::remove_lhs();

	num_lhs=0;
}

//Adapted from CCombinedKernel
void CProductKernel::remove_rhs()
{
	CListElement* current = NULL ;
	CKernel* k=get_first_kernel(current);

	while (k)
	{
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_rhs();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	CKernel::remove_rhs();

	num_rhs=0;
}

//Adapted from CCombinedKernel
void CProductKernel::remove_lhs_and_rhs()
{
	CListElement* current = NULL ;
	CKernel* k=get_first_kernel(current);

	while (k)
	{
		if (k->get_kernel_type() != K_CUSTOM)
			k->remove_lhs_and_rhs();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	CKernel::remove_lhs_and_rhs();

	num_lhs=0;
	num_rhs=0;
}

//Adapted from CCombinedKernel
void CProductKernel::cleanup()
{
	CListElement* current = NULL ;
	CKernel* k=get_first_kernel(current);

	while (k)
	{
		k->cleanup();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	CKernel::cleanup();

	num_lhs=0;
	num_rhs=0;
}

//Adapted from CCombinedKernel
void CProductKernel::list_kernels()
{
	CKernel* k;

	SG_INFO( "BEGIN PRODUCT KERNEL LIST - ");
	this->list_kernel();

	CListElement* current = NULL ;
	k=get_first_kernel(current);
	while (k)
	{
		k->list_kernel();
		SG_UNREF(k);
		k=get_next_kernel(current);
	}
	SG_INFO( "END PRODUCT KERNEL LIST - ");
}

//Adapted from CCombinedKernel
float64_t CProductKernel::compute(int32_t x, int32_t y)
{
	float64_t result=1;
	CListElement* current = NULL ;
	CKernel* k=get_first_kernel(current);
	while (k)
	{
		result *= k->get_combined_kernel_weight() * k->kernel(x,y);
		SG_UNREF(k);
		k=get_next_kernel(current);
	}

	return result;
}

//Adapted from CCombinedKernel

bool CProductKernel::precompute_subkernels()
{
	CKernel* k = get_first_kernel();

	if (!k)
		return false;

	CList* new_kernel_list = new CList(true);

	while(k)
	{
		new_kernel_list->append_element(new CCustomKernel(k));
		SG_UNREF(k);
		k = get_next_kernel();
	}

	SG_UNREF(kernel_list);
	kernel_list=new_kernel_list;
	SG_REF(kernel_list);

	return true;
}

void CProductKernel::init()
{
	initialized=false;

	properties = KP_NONE;
	kernel_list=new CList(true);
	SG_REF(kernel_list);

	SG_ADD((CSGObject**) &kernel_list, "kernel_list", "List of kernels.",
	    MS_AVAILABLE);
	SG_ADD(&initialized, "initialized", "Whether kernel is ready to be used.",
	    MS_NOT_AVAILABLE);
}
