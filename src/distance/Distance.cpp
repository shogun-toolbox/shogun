/*
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Time.h"
#include "base/Parallel.h"

#include "distance/Distance.h"
#include "features/Features.h"

#include <string.h>
#include <unistd.h>

#ifndef WIN32
#include <pthread.h>
#endif

CDistance::CDistance() 
: CSGObject(), precomputed_matrix(NULL), precompute_matrix(false), 
	lhs(NULL), rhs(NULL)
{
}

		
CDistance::CDistance(CFeatures* p_lhs, CFeatures* p_rhs)
:  precomputed_matrix(NULL), precompute_matrix(false), 
	lhs(NULL), rhs(NULL)
{
	init(p_lhs, p_rhs);
}

CDistance::~CDistance()
{
	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;
}

bool CDistance::init(CFeatures* l, CFeatures* r)
{
	//make sure features were indeed supplied
	ASSERT(l);
	ASSERT(r);

	//make sure features are compatible
	ASSERT(l->get_feature_class() == r->get_feature_class());
	ASSERT(l->get_feature_type() == r->get_feature_type());

	lhs=l;
	rhs=r;

	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;

	return true;
}

bool CDistance::load(CHAR* fname)
{
	return false;
}

bool CDistance::save(CHAR* fname)
{
	INT i=0;
	INT num_left=get_lhs()->get_num_vectors();
	INT num_right=get_rhs()->get_num_vectors();
	KERNELCACHE_IDX num_total=num_left*num_right;

	CFile f(fname, 'w', F_DREAL);

    for (INT l=0; l< (INT) num_left && f.is_ok(); l++)
	{
		for (INT r=0; r< (INT) num_right && f.is_ok(); r++)
		{
			if (!(i % (num_total/10+1)))
				SG_PRINT( "%02d%%.", (int) (100.0*i/num_total));
			else if (!(i % (num_total/200+1)))
				SG_PRINT( ".");

			double k=distance(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}

	if (f.is_ok())
		SG_INFO( "distance matrix of size %ld x %ld written \n", num_left, num_right);

    return (f.is_ok());
}

void CDistance::remove_lhs()
{ 
	lhs = NULL;
}

/// takes all necessary steps if the rhs is removed from kernel
void CDistance::remove_rhs()
{
	rhs = NULL;
}


void CDistance::do_precompute_matrix()
{
	INT num_left=get_lhs()->get_num_vectors();
	INT num_right=get_rhs()->get_num_vectors();
	SG_INFO( "precomputing distance matrix (%ix%i)\n", num_left, num_right) ;

	ASSERT(num_left == num_right) ;
	ASSERT(get_lhs()==get_rhs()) ;
	INT num=num_left ;
	
	delete[] precomputed_matrix ;
	precomputed_matrix=new SHORTREAL[num*(num+1)/2] ;
	ASSERT(precomputed_matrix!=NULL) ;

	for (INT i=0; i<num; i++)
	{
		SG_PROGRESS(i*i,0,num*num);
		for (INT j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	SG_PROGRESS(num*num,0,num*num);
	SG_INFO( "\ndone.\n") ;
}

void CDistance::get_distance_matrix(DREAL** dst, INT* m, INT* n)
{
	ASSERT(dst && m && n);

	DREAL* result = NULL;
	CFeatures* f1 = get_lhs();
	CFeatures* f2 = get_rhs();

	if (f1 && f2)
	{
		INT num_vec1=f1->get_num_vectors();
		INT num_vec2=f2->get_num_vectors();
		*m=num_vec1;
		*n=num_vec2;

		LONG total_num = num_vec1 * num_vec2;
		INT num_done = 0;
		SG_DEBUG( "returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		result=new DREAL[total_num];
		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (INT i=0; i<num_vec1; i++)
			{
				for (INT j=i; j<num_vec1; j++)
				{
					double v=distance(i,j);

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
			for (INT i=0; i<num_vec1; i++)
			{
				for (INT j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_PRINT( "done.           \n");
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	*dst=result;
}

SHORTREAL* CDistance::get_distance_matrix_shortreal(int &num_vec1, int &num_vec2, SHORTREAL* target)
{
	SHORTREAL* result = NULL;
	CFeatures* f1 = get_lhs();
	CFeatures* f2 = get_rhs();

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()) )
         SG_ERROR( "distance matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		LONG total_num = num_vec1 * num_vec2;
		int num_done = 0;

		SG_DEBUG( "returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new SHORTREAL[total_num];

		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=i; j<num_vec1; j++)
				{
					double v=distance(i,j);

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
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_PRINT( "done.           \n");
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	return result;
}

DREAL* CDistance::get_distance_matrix_real(int &num_vec1, int &num_vec2, DREAL* target)
{
	DREAL* result = NULL;
	CFeatures* f1 = get_lhs();
	CFeatures* f2 = get_rhs();

	if (f1 && f2)
	{
		if (target && (num_vec1!=f1->get_num_vectors() || num_vec2!=f2->get_num_vectors()) )
         SG_ERROR( "distance matrix does not fit into target\n");

		num_vec1=f1->get_num_vectors();
		num_vec2=f2->get_num_vectors();
		LONG total_num = num_vec1 * num_vec2;
		int num_done = 0;

		SG_DEBUG( "returning distance matrix of size %dx%d\n", num_vec1, num_vec2);

		if (target)
			result=target;
		else
			result=new DREAL[total_num];

		ASSERT(result);

		if ( (f1 == f2) && (num_vec1 == num_vec2) )
		{
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=i; j<num_vec1; j++)
				{
					double v=distance(i,j);

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
			for (int i=0; i<num_vec1; i++)
			{
				for (int j=0; j<num_vec2; j++)
				{
					result[i+j*num_vec1]=distance(i,j) ;

					if (num_done%100000)
						SG_PROGRESS(num_done, 0, total_num-1);

					num_done++;
				}
			}
		}

		SG_PRINT( "done.           \n");
	}
	else
      SG_ERROR( "no features assigned to distance\n");

	return result;
}
