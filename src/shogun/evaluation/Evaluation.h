/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Roman Votyakov, 
 *          Yuyu Zhang
 */

#ifndef EVALUATION_H_
#define EVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/labels/Labels.h>
#include <shogun/base/SGObject.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
class Labels;

/** enum which used to define whether an evaluation measure has to be minimized
 * or maximized
 */
enum EEvaluationDirection
{
	ED_MINIMIZE=0,
	ED_MAXIMIZE=1
};

/** @brief Class Evaluation, a base class for other classes used to evaluate
 * labels, e.g. accuracy of classification or mean squared error of regression.
 *
 * This class provides only interface for evaluation measures.
 */
class Evaluation : public SGObject
{
public:
	/** default constructor */
	Evaluation() : SGObject() { };

	/** destructor */
	virtual ~Evaluation() { };

	/** evaluate labels
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return evaluation result
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)=0;

	/** set absolute indices of labels to be evaluated next used by multitask
	 * evaluations
	 *
	 * @param indices indices
	 */
	virtual void set_indices(SGVector<index_t> indices) { }

	/** @return whether criterion has to be maximized or minimized */
	virtual EEvaluationDirection get_evaluation_direction() const=0;
};
}
#endif /* EVALUATION_H_ */
