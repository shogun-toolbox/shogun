/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang, Saurabh Goyal
 */

#ifndef __CLUSTERINGEVALUATION_H__
#define __CLUSTERINGEVALUATION_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** @brief The base class used to evaluate clustering
 */
class ClusteringEvaluation: public Evaluation
{
public:
	/** constructor */
	ClusteringEvaluation();

	/** destructor */
	virtual ~ClusteringEvaluation() {}

	/** permute the order of the predicted labels to match the ground_truth as good as possible.
	 *
	 * The Munkres assignment algorithm is used to find the best match.
	 * Note this method perform inplace modification on the parameter predicted
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 */
	std::shared_ptr<Labels> best_map(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

	/** evaluate labels
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);
protected:
	/** implementation of label evaluation
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate_impl(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth) = 0;

	/** find number of matches in the two labels sequence.
	 *
	 * For each index i, if l1[i] == m1 and l2[i] == m2, then we get a match.
	 * @param l1 the first label sequence to be matched
	 * @param m1 the first label to match
	 * @param l2 the second label sequence to be matched
	 * @param m2 the second label to match
	 * @return number of matches
	 */
	int32_t find_match_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2);

	/** find number of mismatches in the two labels sequence.
	 * @see find_match_count
	 */
	int32_t find_mismatch_count(SGVector<int32_t> l1, int32_t m1, SGVector<int32_t> l2, int32_t m2);

private:
	// A flag to find best match between predicted labels and the ground truth
	// before evaluation
	bool m_use_best_map;
};

} // namespace shogun

#endif /* end of include guard: __CLUSTERINGEVALUATION_H__ */
