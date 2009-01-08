/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/CombinedDotFeatures.h"
#include "lib/io.h"

CCombinedDotFeatures::CCombinedDotFeatures() : CDotFeatures()
{
	feature_list=new CList<CDotFeatures*>(true);
}

CCombinedDotFeatures::CCombinedDotFeatures(const CCombinedDotFeatures & orig)
: CDotFeatures(orig), num_vectors(orig.num_vectors),
	num_dimensions(orig.num_dimensions)
{
}

CFeatures* CCombinedDotFeatures::duplicate() const
{
	return new CCombinedDotFeatures(*this);
}

CCombinedDotFeatures::~CCombinedDotFeatures()
{
	delete feature_list;
}

void CCombinedDotFeatures::list_feature_objs()
{
	SG_INFO( "BEGIN COMBINED FEATURES LIST - ");
	this->list_feature_obj();

	CListElement<CDotFeatures*> * current = NULL ;
	CDotFeatures* f=get_first_feature_obj(current);

	while (f)
	{
		f->list_feature_obj();
		f=get_next_feature_obj(current);
	}

	SG_INFO( "END COMBINED FEATURES LIST - ");
}

int32_t CCombinedDotFeatures::get_dim_feature_space()
{
	CListElement<CDotFeatures*> * current = NULL ;
	CDotFeatures* f=get_first_feature_obj(current);

	int32_t dim=0;

	while (f)
	{
		dim+= f->get_dim_feature_space();
		f=get_next_feature_obj(current);
	}

	return dim;
}

float64_t CCombinedDotFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	float64_t result=0;

	CListElement<CDotFeatures*> * current = NULL ;
	CDotFeatures* f=get_first_feature_obj(current);

	while (f)
	{
		result += f->dot(vec_idx1, vec_idx2);
		f=get_next_feature_obj(current);
	}

	return result;
}

float64_t CCombinedDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	float64_t result=0;

	CListElement<CDotFeatures*> * current = NULL ;
	CDotFeatures* f=get_first_feature_obj(current);
	uint32_t offs=0;

	while (f)
	{
		result += f->dense_dot(vec_idx1, vec2+offs, vec2_len);
		offs += f->get_dim_feature_space();
		f=get_next_feature_obj(current);
	}

	return result;
}

void CCombinedDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	CListElement<CDotFeatures*> * current = NULL ;
	CDotFeatures* f=get_first_feature_obj(current);
	uint32_t offs=0;

	while (f)
	{
		f->add_to_dense_vec(alpha, vec_idx1, vec2+offs, vec2_len, abs_val);
		offs += f->get_dim_feature_space();
		f=get_next_feature_obj(current);
	}
}
