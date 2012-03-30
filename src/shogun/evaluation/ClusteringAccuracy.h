#ifndef __CLUSTERINGACCURACY_H__
#define __CLUSTERINGACCURACY_H__

#include <shogun/evaluation/ClusteringEvaluation.h>

namespace shogun
{


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
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth)
    {
        int correct = 0;
        for (int i = predicted->get_num_labels()-1; i >= 0; --i) {
            if (predicted->get_int_label(i) == ground_truth->get_int_label(i))
                correct++;
        }
        return float64_t(correct)/predicted->get_num_labels();
    }

	/** @return whether criterium has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction()
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

