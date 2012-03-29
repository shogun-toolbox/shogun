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

    /** permute predicted labels to match the ground_truth as good as possible
     * The Munkres assignment algorithm is used to find the best match.
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
