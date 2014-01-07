/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef __CLUSTERINGEVALUATION_H__
#define __CLUSTERINGEVALUATION_H__

#include <vector>

#include <evaluation/Evaluation.h>
#include <labels/Labels.h>

namespace shogun
{

/** @brief The base class used to evaluate clustering
 */
class CClusteringEvaluation: public CEvaluation
{
public:
	/** constructor */
	CClusteringEvaluation(): CEvaluation() {}

	/** destructor */
	virtual ~CClusteringEvaluation() {}

	/** permute the order of the predicted labels to match the ground_truth as good as possible.
	 *
	 * The Munkres assignment algorithm is used to find the best match.
	 * Note this method perform inplace modification on the parameter predicted
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 */
	void best_map(CLabels* predicted, CLabels* ground_truth);

	/** evaluate labels
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth) = 0;
protected:
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
};

} // namespace shogun

#endif /* end of include guard: __CLUSTERINGEVALUATION_H__ */
