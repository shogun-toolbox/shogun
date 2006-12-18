/*
 * this program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Written (W) 2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "lib/Time.h"
#include "lib/Parallel.h"

#include "distance/Distance.h"
#include "features/Features.h"

#include <string.h>
#include <unistd.h>

CDistance::CDistance() 
: precomputed_matrix(NULL), precompute_matrix(false), 
	lhs(NULL), rhs(NULL)
{
}

		
CDistance::CDistance(CFeatures* p_lhs, CFeatures* p_rhs)
:  precomputed_matrix(NULL), precompute_matrix(false), 
	lhs(NULL), rhs(NULL)
{
	init(p_lhs, p_rhs, true);
}

CDistance::~CDistance()
{
	delete[] precomputed_matrix ;
	precomputed_matrix=NULL ;
}

bool CDistance::init(CFeatures* l, CFeatures* r, bool do_init)
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
				CIO::message(M_MESSAGEONLY, "%02d%%.", (int) (100.0*i/num_total));
			else if (!(i % (num_total/200+1)))
				CIO::message(M_MESSAGEONLY, ".");

			double k=distance(l,r);
			f.save_real_data(&k, 1);

			i++;
		}
	}

	if (f.is_ok())
		CIO::message(M_INFO, "distance matrix of size %ld x %ld written \n", num_left, num_right);

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

	CIO::message(M_INFO, "precomputing distance matrix (%ix%i)\n", num_left, num_right) ;

	ASSERT(num_left == num_right) ;
	ASSERT(get_lhs()==get_rhs()) ;
	INT num=num_left ;
	
	delete[] precomputed_matrix ;
	precomputed_matrix=new SHORTREAL[num*(num+1)/2] ;

	ASSERT(precomputed_matrix!=NULL) ;

	for (INT i=0; i<num; i++)
	{
		CIO::progress(i*i,0,num*num);
		for (INT j=0; j<=i; j++)
			precomputed_matrix[i*(i+1)/2+j] = compute(i,j) ;
	}

	CIO::progress(num*num,0,num*num);
	CIO::message(M_INFO, "\ndone.\n") ;
}
