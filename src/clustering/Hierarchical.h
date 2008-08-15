/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HIERARCHICAL_H__
#define _HIERARCHICAL_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/RealFeatures.h"
#include "distance/Distance.h"
#include "distance/DistanceMachine.h"

class CDistanceMachine;

/** Agglomerative hierarchical single linkage clustering. Starting with each
 * object being assigned to its own cluster clusters are iteratively merged.
 * Here the clusters are merged whose elements have minimum distance, i.e.
 * the clusters A and B that obtain
 *
 * \f[
 * \min\{d({\bf x},{\bf x'}): {\bf x}\in {\cal A},{\bf x'}\in {\cal B}\}
 * \f]
 *
 * are merged.
 *
 * cf e.g. http://en.wikipedia.org/wiki/Data_clustering*/
class CHierarchical : public CDistanceMachine
{
	public:
		/** default constructor */
		CHierarchical();

		/** constructor
		 *
		 * @param merges the merges
		 * @param d distance
		 */
		CHierarchical(INT merges, CDistance* d);
		virtual ~CHierarchical();

		/** get classifier type
		 *
		 * @return classifier type HIERARCHICAL
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_HIERARCHICAL; }

		/** train distance machine
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** load distance machine from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save distance machine to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** set merges
		 *
		 * @param m new merges
		 */
		inline void set_merges(INT m)
		{
			ASSERT(m>0);
			merges=m;
		}

		/** get merges
		 *
		 * @return merges
		 */
		inline INT get_merges()
		{
			return merges;
		}

		/** get assignment
		 *
		 * @param assign current assignment is stored in here
		 * @param num number of assignments is stored in here
		 */
		inline void get_assignment(INT*& assign, INT& num)
		{
			assign=assignment;
			num=table_size;
		}

		/** get merge distance
		 *
		 * @param dist current merge distance is stored in here
		 * @param num number of merge distances is stored in here
		 */
		inline void get_merge_distance(DREAL*& dist, INT& num)
		{
			dist=merge_distance;
			num=merges;
		}

		/** get merge distances (swig compatible)
		 *
		 * @param dist current merge distances is stored in here
		 * @param num number of merge distances is stored in here
		 */
		inline void get_merge_distances(DREAL** dist, INT* num)
		{
			size_t sz=sizeof(*merge_distance)*merges;
			*dist=(DREAL*) malloc(sz);
			ASSERT(*dist);

			memcpy(*dist, merge_distance, sz);
			*num=merges;
		}

		/** get pairs
		 *
		 * @param tuples current pairs are stored in here
		 * @param rows number of rows is stored in here
		 * @param num number of pairs is stored in here
		 */
		inline void get_pairs(INT*& tuples, INT& rows, INT& num)
		{
			tuples=pairs;
			rows=2;
			num=merges;
		}

		/** get cluster pairs (swig compatible)
		 *
		 * @param tuples current pairs are stored in here
		 * @param rows number of rows is stored in here
		 * @param num number of pairs is stored in here
		 */
		inline void get_cluster_pairs(INT** tuples, INT* rows, INT* num)
		{
			*rows=2;
			size_t sz=sizeof(*pairs)*(*rows)*merges;
			*tuples=(INT*) malloc(sz);
			ASSERT(*tuples);

			memcpy(*tuples, pairs, sz);
			*num=merges;
		}

	protected:
		/// the number of merges in hierarchical clustering
		INT merges;

		/// number of dimensions
		INT dimensions;

		/// size of assignment table
		INT assignment_size;

		/// cluster assignment for the num_points
		INT* assignment;

		/// size of the below tables
		INT table_size;

		/// tuples of i/j
		INT* pairs;

		/// distance at which pair i/j was added
		DREAL* merge_distance;
};
#endif
