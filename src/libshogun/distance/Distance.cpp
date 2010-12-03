/*
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Time.h"
#include "base/Parallel.h"
#include "base/Parameter.h"

#include "distance/Distance.h"
#include "features/Features.h"

#include <string.h>
#include <unistd.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

CDistance::CDistance() : CSGObject()
{
	init();
}

		
CDistance::CDistance(CFeatures* p_lhs, CFeatures* p_rhs) : CSGObject()
{
	init();
	init(p_lhs, p_rhs);
}

CDistance::~CDistance()
{
	delete[] precomputed_matrix;
	precomputed_matrix=NULL;

	remove_lhs_and_rhs();
}

bool CDistance::init(CFeatures* l, CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(l);
	ASSERT(r);

	//make sure features are compatible
	ASSERT(l->get_feature_class()==r->get_feature_class());
	ASSERT(l->get_feature_type()==r->get_feature_type());

	//remove references to previous features
	remove_lhs_and_rhs();

    //increase reference counts
    SG_REF(l);
	SG_REF(r);

	lhs=l;
	rhs=r;

	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;

	return true;
}

void CDistance::load(CFile* loader)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CDistance::save(CFile* writer)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
}

void CDistance::remove_lhs_and_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;

	SG_UNREF(lhs);
	lhs = NULL;
}

void CDistance::remove_lhs()
{ 
	SG_UNREF(lhs);
	lhs = NULL;
}

/// takes all necessary steps if the rhs is removed from kernel
void CDistance::remove_rhs()
{
	SG_UNREF(rhs);
	rhs = NULL;
}

CFeatures* CDistance::replace_rhs(CFeatures* r)
{
     //make sure features were indeed supplied
     ASSERT(r);

     //make sure features are compatible
     ASSERT(lhs->get_feature_class()==r->get_feature_class());
     ASSERT(lhs->get_feature_type()==r->get_feature_type());

     //remove references to previous rhs features
     CFeatures* tmp=rhs;
     
     rhs=r;

     delete[] precomputed_matrix ;
     precomputed_matrix=NULL ;

	 // return old features including reference count
     return tmp;
}

void CDistance::do_precompute_matrix()
{
	int32_t num_left=lhs->get_num_vectors();
	int32_t num_right=rhs->get_num_vectors();
	SG_INFO( "precomputing distance matrix (%ix%i)\n", num_left, num_right) ;

	ASSERT(num_left==num_right);
	ASSERT(lhs==rhs);
	int32_t num=num_left;
	
	delete[] precomputed_matrix;
	precomputed_matrix=new float32_t[num*(num+1)/2];

	for (int32_t i=0; i<num; i++)
	{
		SG_PROGRESS(i*i,0,num*num);
		for (int32_t j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	SG_PROGRESS(num*num,0,num*num);
	SG_DONE();
}

void CDistance::get_distance_matrix(float64_t** dst, int32_t* m, int32_t* n)
{
	ASSERT(dst && m && n);

	float64_t* result = NULL;
	CFeatures* f1 = lhs;
	CFeatures* f2 = rhs;

	if (f1 && f2)
	{
		int32_t num_vec1=f1->get_num_vectors();
		int32_t num_vec2=f2->get_num_vectors();
		*m=num_vec1;
		*n=num_vec2;

		int64_t total_num=num_vec1*num_vec2;
		int32_t num_done=0;
		SG_DEBUG("returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		result=(float64_t*) malloc(total_num*sizeof(float64_t));
		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=distance(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	*dst=result;
}

float32_t* CDistance::get_distance_matrix_shortreal(
	int32_t &num_vec1, int32_t &num_vec2, float32_t* target)
{
	float32_t* result = NULL;
	CFeatures* f1 = lhs;
	CFeatures* f2 = rhs;

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()))
			SG_ERROR("distance matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		int64_t total_num=num_vec1*num_vec2;
		int32_t num_done=0;

		SG_DEBUG("returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new float32_t[total_num];

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=distance(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	return result;
}

float64_t* CDistance::get_distance_matrix_real(
	int32_t &num_vec1, int32_t &num_vec2, float64_t* target)
{
	float64_t* result = NULL;
	CFeatures* f1 = lhs;
	CFeatures* f2 = rhs;

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()))
			SG_ERROR("distance matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		int64_t total_num=num_vec1*num_vec2;
		int32_t num_done=0;

		SG_DEBUG("returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new float64_t[total_num];

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=i; j<num_vec1; j++)
				{
					float64_t v=distance(i,j);

					result[i+j*num_vec1]=v;
					result[j+i*num_vec1]=v;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					if (i!=j)
						num_done+=2;
					else
						num_done+=1;
				}
			}
		}
		else
		{
			for (int32_t i=0; i<num_vec1; i++)
			{
				for (int32_t j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_DONE();
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	return result;
}

void CDistance::init()
{
	precomputed_matrix = NULL;
	precompute_matrix = false;
	lhs = NULL;
	rhs = NULL;

	m_parameters->add((CSGObject**) &lhs, "lhs",
					  "Feature vectors to occur on left hand side.");
	m_parameters->add((CSGObject**) &rhs, "rhs",
					  "Feature vectors to occur on right hand side.");
}
