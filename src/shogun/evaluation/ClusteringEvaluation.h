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

#include <shogun/evaluation/Evaluation.h>
#include <shogun/features/Labels.h>

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
};

} // namespace shogun

#endif /* end of include guard: __CLUSTERINGEVALUATION_H__ */
