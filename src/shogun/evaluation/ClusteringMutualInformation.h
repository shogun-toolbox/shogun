/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef __CLUSTERINGMUTUALINFORMATION_H__
#define __CLUSTERINGMUTUALINFORMATION_H__

#include <evaluation/ClusteringEvaluation.h>

namespace shogun
{

/** @brief clustering (normalized) mutual information
 */
class CClusteringMutualInformation: public CClusteringEvaluation
{
public:
	/** constructor */
	CClusteringMutualInformation(): CClusteringEvaluation() {}

	/** destructor */
	virtual ~CClusteringMutualInformation() {}

	/** evaluate labels
	 * Make sure to call CClusteringEvaluation::best_map to map the predicted label
	 * before calculating mutual information.
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** @return whether criterium has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MINIMIZE;
	}

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const
	{
		return "ClusteringMutualInformation";
	}
};

}

#endif /* end of include guard: __CLUSTERINGMUTUALINFORMATION_H__ */
