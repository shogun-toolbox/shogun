/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HIERARCHICAL_H__
#define _HIERARCHICAL_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "distance/Distance.h"
#include "machine/DistanceMachine.h"

namespace shogun
{
class CDistanceMachine;

/** @brief Agglomerative hierarchical single linkage clustering.
 *
 * Starting with each object being assigned to its own cluster clusters are
 * iteratively merged.  Here the clusters are merged whose elements have
 * minimum distance, i.e.  the clusters A and B that obtain
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
		CHierarchical(int32_t merges, CDistance* d);
		virtual ~CHierarchical();

		/** get classifier type
		 *
		 * @return classifier type HIERARCHICAL
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_HIERARCHICAL; }

		/** estimate hierarchical clustering
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

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
		inline void set_merges(int32_t m)
		{
			ASSERT(m>0);
			merges=m;
		}

		/** get merges
		 *
		 * @return merges
		 */
		inline int32_t get_merges()
		{
			return merges;
		}

		/** get assignment
		 *
		 * @param assign current assignment is stored in here
		 * @param num number of assignments is stored in here
		 */
		inline void get_assignment(int32_t*& assign, int32_t& num)
		{
			assign=assignment;
			num=table_size;
		}

		/** get merge distance
		 *
		 * @param dist current merge distance is stored in here
		 * @param num number of merge distances is stored in here
		 */
		inline void get_merge_distance(float64_t*& dist, int32_t& num)
		{
			dist=merge_distance;
			num=merges;
		}

		/** get merge distances (swig compatible)
		 *
		 * @param dist current merge distances are stored in here
		 * @param num number of merge distances are stored in here
		 */
		inline void get_merge_distances(float64_t** dist, int32_t* num)
		{
			size_t sz=sizeof(*merge_distance)*merges;
			*dist=(float64_t*) SG_MALLOC(sz);
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
		inline void get_pairs(int32_t*& tuples, int32_t& rows, int32_t& num)
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
		inline void get_cluster_pairs(
			int32_t** tuples, int32_t* rows, int32_t* num)
		{
			*rows=2;
			size_t sz=sizeof(*pairs)*(*rows)*merges;
			*tuples=(int32_t*) SG_MALLOC(sz);
			ASSERT(*tuples);

			memcpy(*tuples, pairs, sz);
			*num=merges;
		}

		/** classify objects using the currently set features
		 *
		 * @return classified labels
		 */
		virtual CLabels* apply()
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data)
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "Hierarchical"; }

	protected:
		/// the number of merges in hierarchical clustering
		int32_t merges;

		/// number of dimensions
		int32_t dimensions;

		/// size of assignment table
		int32_t assignment_size;

		/// cluster assignment for the num_points
		int32_t* assignment;

		/// size of the below tables
		int32_t table_size;

		/// tuples of i/j
		int32_t* pairs;

		/// distance at which pair i/j was added
		float64_t* merge_distance;
};
}
#endif
