/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * 
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "clustering/Hierarchical.h"
#include "distance/Distance.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"

#ifndef WIN32
#include <pthread.h>
#endif

CHierarchical::CHierarchical(): merges(3), dimensions(0), assignment(NULL),
	table_size(0), pairs(NULL), merge_distance(NULL)
{
}

CHierarchical::CHierarchical(INT merges_, CDistance* d): merges(merges_), dimensions(0), assignment(NULL),
	table_size(0), pairs(NULL), merge_distance(NULL)
{
	set_distance(d);
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

	const INT num_pairs=num*(num-1)/2;

	merge_distance=new DREAL[num];
	ASSERT(merge_distance);
	CMath::fill_vector(merge_distance, num, -1.0);

	assignment=new INT[num];
	ASSERT(assignment);
	CMath::range_fill_vector(assignment, num);

	pairs=new INT[2*num];
	ASSERT(pairs);
	CMath::fill_vector(pairs, 2*num, -1);

	pair* index=new pair[num_pairs];
	ASSERT(index);
	DREAL* distances=new DREAL[num_pairs];
	ASSERT(distances);

	INT offs=0;
	for (INT i=0; i<num; i++)
	{
		for (INT j=i+1; j<num; j++)
		{
			distances[offs]=d->distance(i,j);
			index[offs].idx1=i;
			index[offs].idx2=j;
			offs++;					//offs=i*(i+1)/2+j
		}
		SG_PROGRESS(i, 0, num-1);
	}

	CMath::qsort_index<DREAL,pair>(distances, index, (num-1)*num/2);
	//CMath::display_vector(distances, (num-1)*num/2, "dists");

	INT k=-1;
	INT l=0;
	for (; l<num && (num-l)>=merges && k<num_pairs-1; l++)
	{
		while (k<num_pairs-1)
		{
			k++;

			INT i=index[k].idx1;
			INT j=index[k].idx2;
			INT c1=assignment[i];
			INT c2=assignment[j];

			if (c1==c2)
				continue;
			
			SG_PROGRESS(k, 0, num_pairs-1);

			if (c1<c2)
			{
				pairs[2*l]=c1;
				pairs[2*l+1]=c2;
			}
			else
			{
				pairs[2*l]=c2;
				pairs[2*l+1]=c1;
			}
			merge_distance[l]=distances[k];

			INT c=num+l;
			for (INT m=0; m<num; m++)
			{
				if (assignment[m] == c1 || assignment[m] == c2)
					assignment[m] = c;
			}
#ifdef DEBUG_HIERARCHICAL
			SG_PRINT("l=%04i i=%04i j=%04i c1=%+04d c2=%+04d c=%+04d dist=%6.6f\n", l,i,j, c1,c2,c, merge_distance[l]);
#endif
			break;
		}
	}

	assignment_size=num;
	table_size=l-1;
	ASSERT(table_size>0);
	delete[] distances;
	delete[] index;

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
