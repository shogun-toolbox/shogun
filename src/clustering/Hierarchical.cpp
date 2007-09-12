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

CHierarchical::CHierarchical(): merges(3), dimensions(0), table_size(0), assignment(NULL), pairs(NULL)
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

	assignment=new DREAL[2*num];
	ASSERT(assignment);
	CMath::fill_vector(assignment, 2*num, -1.0);

	pairs=new INT[2*num];
	ASSERT(pairs);
	CMath::fill_vector(pairs, 2*num, -1);

	INT clusters=-1;
	INT k=0;

	for (k=0; k<num && clusters<merges; k++)
	{
		INT m;
		DREAL cur_dist=+1e+30;
		INT best_i=-1;
		INT best_j=-1;

		for (INT i=0; i<num; i++)
		{
			for (INT j=i+1; j<num; j++)
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

		INT c1=-1;
		bool found1=false;
		bool found2=false;

		for (m=0; m<k; m+=2)
		{
			if ((pairs[m]==best_i) || (pairs[m]==best_j) ||
					(pairs[m+1]==best_i) || (pairs[m+1]==best_j))
			{
				c1=(INT) assignment[m+1];
				found1=true;
				break;
			}
		}

		INT c2=-1;
		for (; m<k; m+=2)
		{
			if ((pairs[m]==best_i) || (pairs[m]==best_j) ||
					(pairs[m+1]==best_i) || (pairs[m+1]==best_j))
			{
				c2=(INT) assignment[m+1];
				found2=true;
				break;
			}
		}

		INT c=-1;
		if (!found1 && !found2)
		{
			clusters++;
			c=clusters;
		}
		else if (found1 && found2)
			c=CMath::min(c1,c2);
		else if (found1)
			c=c1;
		else if (found2)
			c=c2;
		else
			SG_ERROR("internal error");

		pairs[2*k]=best_i;
		pairs[2*k+1]=best_j;
		assignment[2*k]=cur_dist;
		assignment[2*k+1]=c;

		for (m=best_j; m<num && k<num; m++)
		{
			DREAL dd=d->distance(best_i,m);
			if (dd==cur_dist)
			{
				pairs[2*k]=best_i;
				pairs[2*k+1]=m;
				assignment[2*k]=cur_dist;
				assignment[2*k+1]=c;
			}
		}

		if (found1 && found2)
		{
			INT cmax=CMath::max(c1,c2);
			for (m=0; m<k; m+=2)
			{
				if (assignment[m+1] == cmax)
					assignment[m+1] = c;
			}
		}

		old_dist=cur_dist;
	}

	table_size=k;

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
