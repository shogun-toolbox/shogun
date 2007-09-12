/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * 
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "clustering/Hierarchical.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

CHierarchical::CHierarchical(): k(3), dimensions(0), assignment(NULL), pairs(NULL)
{
}

CHierarchical::~CHierarchical()
{
}

bool CHierarchical::train()
{
	CDistance* d=CDistanceMachine::get_distance();
	ASSERT(d);

	CFeatures* lhs = d->get_lhs();
	ASSERT(lhs);

	INT num=lhs->get_num_vectors();
	ASSERT(num>0);

	DREAL old_dist=-1e+30;
	INT num_clusters=-1;
	INT num_elements=0;

	assignment=new DREAL[2*num];
	CMath::fill_vector(assignment, 2*num, -1.0);
	pairs=new INT[2*num];
	CMath::fill_vector(pairs, 2*num, -1);

	INT l=0;
	while (num_clusters<=k && num_elements<num)
	{
		DREAL cur_dist=+1e+30;
		INT best_i=-1;
		INT best_j=-1;

		for (INT i=0; i<num; i++)
		{
			for (INT j=0; j<num; j++)
			{
				DREAL dd=d->distance(i,j);
				if (dd>old_dist && dd<cur_dist)
				{
					cur_dist=dd;
					best_i=i;
					best_j=j;
				}
			}
		}

		INT cluster=-1;
		bool found=false;

		for (INT m=0; m<l; m+=2)
		{
			if ((pairs[m]==best_i) || (pairs[m]==best_j) ||
					(pairs[m+1]==best_i) || (pairs[m+1]==best_j))
			{
				cluster=assignment[m+1];
				found=true;
				break;
			}
		}

		if (!found)
		{
			num_clusters++;
			cluster=num_clusters;
		}

		pairs[l]=best_i;
		pairs[l+1]=best_j;
		assignment[l]=cur_dist;
		assignment[l+1]=cluster;
		l+=2;

		for (INT m=best_j; m<num; m++)
		{
			DREAL dd=d->distance(best_i,m);
			if (dd==cur_dist)
			{
				pairs[l]=best_i;
				pairs[l+1]=m;
				assignment[l]=cur_dist;
				assignment[l+1]=cluster;
				l+=2;
			}
		}

		old_dist=cur_dist;
	}

	return true;
}

CLabels* CHierarchical::classify(CLabels* output)
{
	return NULL;
}

bool CHierarchical::load(FILE* srcfile)
{
	return false;
}

bool CHierarchical::save(FILE* dstfile)
{
	return false;
}
