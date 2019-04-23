/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Yuyu Zhang
 */

#ifndef BINARYCLASSEVALUATION_H_
#define BINARYCLASSEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/BinaryLabels.h>

namespace shogun
{

class Labels;

/** @brief The class TwoClassEvaluation,
 * a base class used to evaluate binary classification
 * labels.
 *
 */
class BinaryClassEvaluation: public Evaluation
{

public:

	/** constructor */
	BinaryClassEvaluation() : Evaluation() {};

	/** destructor */
	virtual ~BinaryClassEvaluation() {};

	/** evaluate labels
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth) = 0;
};

}


#endif /* BINARYCLASSEVALUATION_H_ */
