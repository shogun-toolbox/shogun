/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann, Yuyu Zhang, 
 *          Bjoern Esser, Saurabh Goyal
 */

#ifndef _HIERARCHICAL_H__
#define _HIERARCHICAL_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
class DistanceMachine;

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
class Hierarchical : public DistanceMachine
{
	public:
		/** default constructor */
		Hierarchical();

		/** constructor
		 *
		 * @param merges the merges
		 * @param d distance
		 */
		Hierarchical(int32_t merges, std::shared_ptr<Distance> d);
		virtual ~Hierarchical();

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

		virtual bool train_require_labels() const
		{
			return false;
		}

	protected:
		/** estimate hierarchical clustering
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);

	private:
		/** Initialize attributes */
		void init();

		/** Register all parameters (aka this class' attributes) */
		void register_parameters();

	protected:
		/// the number of merges in hierarchical clustering
		int32_t merges;

		/// number of dimensions
		int32_t dimensions;

		/// cluster assignment for the num_points
		int32_t* assignment;
		int32_t assignment_len;

		/// size of the below tables
		int32_t table_size;

		/// tuples of i/j
		int32_t* pairs;
		int32_t pairs_len;

		/// distance at which pair i/j was added
		float64_t* merge_distance;
		int32_t merge_distance_len;
};
}
#endif
