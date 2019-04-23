/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Roman Votyakov, Yuyu Zhang
 */

#ifndef __CLUSTERINGMUTUALINFORMATION_H__
#define __CLUSTERINGMUTUALINFORMATION_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/ClusteringEvaluation.h>

namespace shogun
{

/** @brief clustering (normalized) mutual information
 */
class ClusteringMutualInformation: public ClusteringEvaluation
{
public:
	/** constructor */
	ClusteringMutualInformation(): ClusteringEvaluation() {}

	/** destructor */
	virtual ~ClusteringMutualInformation() {}

	/** evaluate labels
	 * Make sure to call ClusteringEvaluation::best_map to map the predicted label
	 * before calculating mutual information.
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

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
