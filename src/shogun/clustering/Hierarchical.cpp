/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <clustering/Hierarchical.h>
#include <distance/Distance.h>
#include <labels/Labels.h>
#include <features/Features.h>
#include <mathematics/Math.h>
#include <base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct pair
{
	/** index 1 */
	int32_t idx1;
	/** index 2 */
	int32_t idx2;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CHierarchical::CHierarchical()
: CDistanceMachine(), merges(3), dimensions(0), assignment(NULL),
	table_size(0), pairs(NULL), merge_distance(NULL)
{
}

CHierarchical::CHierarchical(int32_t merges_, CDistance* d)
: CDistanceMachine(), merges(merges_), dimensions(0), assignment(NULL),
	table_size(0), pairs(NULL), merge_distance(NULL)
{
	set_distance(d);
}

CHierarchical::~CHierarchical()
{
	SG_FREE(merge_distance);
	SG_FREE(assignment);
	SG_FREE(pairs);
}

EMachineType CHierarchical::get_classifier_type()
{
	return CT_HIERARCHICAL;
}

bool CHierarchical::train_machine(CFeatures* data)
{
	ASSERT(distance)

	if (data)
		distance->init(data, data);

	CFeatures* lhs=distance->get_lhs();
	ASSERT(lhs)

	int32_t num=lhs->get_num_vectors();
	ASSERT(num>0)

	const int32_t num_pairs=num*(num-1)/2;

	SG_FREE(merge_distance);
	merge_distance=SG_MALLOC(float64_t, num);
	SGVector<float64_t>::fill_vector(merge_distance, num, -1.0);

	SG_FREE(assignment);
	assignment=SG_MALLOC(int32_t, num);
	SGVector<int32_t>::range_fill_vector(assignment, num);

	SG_FREE(pairs);
	pairs=SG_MALLOC(int32_t, 2*num);
	SGVector<int32_t>::fill_vector(pairs, 2*num, -1);

	pair* index=SG_MALLOC(pair, num_pairs);
	float64_t* distances=SG_MALLOC(float64_t, num_pairs);

	int32_t offs=0;
	for (int32_t i=0; i<num; i++)
	{
		for (int32_t j=i+1; j<num; j++)
		{
			distances[offs]=distance->distance(i,j);
			index[offs].idx1=i;
			index[offs].idx2=j;
			offs++;					//offs=i*(i+1)/2+j
		}
		SG_PROGRESS(i, 0, num-1)
	}

	CMath::qsort_index<float64_t,pair>(distances, index, (num-1)*num/2);
	//CMath::display_vector(distances, (num-1)*num/2, "dists");

	int32_t k=-1;
	int32_t l=0;
	for (; l<num && (num-l)>=merges && k<num_pairs-1; l++)
	{
		while (k<num_pairs-1)
		{
			k++;

			int32_t i=index[k].idx1;
			int32_t j=index[k].idx2;
			int32_t c1=assignment[i];
			int32_t c2=assignment[j];

			if (c1==c2)
				continue;

			SG_PROGRESS(k, 0, num_pairs-1)

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

			int32_t c=num+l;
			for (int32_t m=0; m<num; m++)
			{
				if (assignment[m] == c1 || assignment[m] == c2)
					assignment[m] = c;
			}
#ifdef DEBUG_HIERARCHICAL
			SG_PRINT("l=%04i i=%04i j=%04i c1=%+04d c2=%+04d c=%+04d dist=%6.6f\n", l,i,j, c1,c2,c, merge_distance[l])
#endif
			break;
		}
	}

	assignment_size=num;
	table_size=l-1;
	ASSERT(table_size>0)
	SG_FREE(distances);
	SG_FREE(index);
	SG_UNREF(lhs)

	return true;
}

bool CHierarchical::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CHierarchical::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}


int32_t CHierarchical::get_merges()
{
	return merges;
}

SGVector<int32_t> CHierarchical::get_assignment()
{
	return SGVector<int32_t>(assignment,table_size, false);
}

SGVector<float64_t> CHierarchical::get_merge_distances()
{
	return SGVector<float64_t>(merge_distance,merges, false);
}

SGMatrix<int32_t> CHierarchical::get_cluster_pairs()
{
	return SGMatrix<int32_t>(pairs,2,merges, false);
}


void CHierarchical::store_model_features()
{
	/* TODO. Currently does nothing since apply methods are not implemented. */
}
