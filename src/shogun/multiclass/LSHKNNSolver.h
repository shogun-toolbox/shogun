/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef LSHSOLVER_H__
#define LSHSOLVER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/distance/Distance.h>
#include <shogun/multiclass/KNNSolver.h>

namespace shogun
{

/**
 * LSH solver. It uses LSH (short for Locality-sensitive hashing) to do the nearest neighbour computation.
 * For more information, see https://en.wikipedia.org/wiki/Locality-sensitive_hashing.
 *
 */
class LSHKNNSolver : public KNNSolver
{
	public:
		/** default constructor */
		LSHKNNSolver() : KNNSolver()
		{
			init();
		}

		/** deconstructor */
		virtual ~LSHKNNSolver() { /* nothing to do */ }

		/** constructor
		 *
		 * @param k k
		 * @param q m_q
		 * @param num_classes m_num_classes
		 * @param min_label m_min_label
		 * @param train_labels m_train_labels
		 * @param lsh_l m_lsh_l
		 * @param lsh_t m_lsh_t
		 */
		LSHKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels, const int32_t lsh_l, const int32_t lsh_t);

		virtual std::shared_ptr<MulticlassLabels> classify_objects(std::shared_ptr<Distance> d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const;

		virtual SGVector<int32_t> classify_objects_k(std::shared_ptr<Distance> d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const;

		/** @return object name */
		const char* get_name() const { return "LSHKNNSolver"; }

	private:
		void init()
		{
			m_lsh_l=0;
			m_lsh_t=0;
		}

		template<typename PointType, typename FeatureType>
		std::shared_ptr<MulticlassLabels> classify_objects(FeatureType* lhs, FeatureType* query_features, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const;

	protected:
		/* Number of hash tables for LSH */
		int32_t m_lsh_l;

		/* Number of probes per query for LSH */
		int32_t m_lsh_t;

};
}

#endif
