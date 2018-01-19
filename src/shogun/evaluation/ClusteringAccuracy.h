/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Roman Votyakov, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef __CLUSTERINGACCURACY_H__
#define __CLUSTERINGACCURACY_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/ClusteringEvaluation.h>

namespace shogun
{

/** @brief clustering accuracy
 */
class CClusteringAccuracy: public CClusteringEvaluation
{
public:
	/** constructor */
	CClusteringAccuracy(): CClusteringEvaluation() {}

	/** destructor */
	virtual ~CClusteringAccuracy() {}

	/** evaluate labels
	 * Make sure to call CClusteringEvaluation::best_map to map the predicted label
	 * before calculating accuracy.
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
		return "ClusteringAccuracy";
	}
};

} // namespace shogun

#endif /* end of include guard: __CLUSTERINGACCURACY_H__ */
