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

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

using namespace distance;

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
		CHierarchical(int32_t merges, distance::CDistance* d);
		virtual ~CHierarchical();

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS);

		/** get classifier type
		 *
		 * @return classifier type HIERARCHICAL
		 */
		virtual EMachineType get_classifier_type();

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
			ASSERT(m>0)
			merges=m;
		}

		/** get merges
		 *
		 * @return merges
		 */
		int32_t get_merges();

		/** get assignment
		 *
		 */
		SGVector<int32_t> get_assignment();

		/** get merge distance
		 *
		 */
		SGVector<float64_t> get_merge_distances();

		/** get cluster pairs
		 *
		 */
		SGMatrix<int32_t> get_cluster_pairs();

		/** @return object name */
		virtual const char* get_name() const { return "Hierarchical"; }

	protected:
		/** estimate hierarchical clustering
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** TODO: Ensures cluster centers are in lhs of underlying distance
		 * Currently: does nothing.
		 * */
		virtual void store_model_features();

		virtual bool train_require_labels() const { return false; }

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
