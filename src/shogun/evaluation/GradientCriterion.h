/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Heiko Strathmann, Yuyu Zhang, 
 *          Sergey Lisitsyn, Soeren Sonnenburg
 */

#ifndef CGRADIENTCRITERION_H_
#define CGRADIENTCRITERION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

/** @brief Simple class which specifies the direction of gradient search.
 *
 * Does not provide any label evaluation measure, however.
 */
class GradientCriterion : public Evaluation
{
public:
	/** default constructor */
	GradientCriterion() : Evaluation() { m_direction=ED_MINIMIZE; }

	~GradientCriterion() override { }

	/** evaluate labels (not really used in this class).
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return evaluation result
	 */
	float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth) override
	{
		return 0.0;
	}

	/** @return whether criterion has to be maximized or minimized */
	EEvaluationDirection get_evaluation_direction() const override
	{
		return m_direction;
	}

	/** set the evaluation direction
	 *
	 * @param direction evaluation direction to be set
	 */
	virtual void set_evaluation_direction(EEvaluationDirection direction)
	{
		m_direction=direction;
	}

	/** returns name of evaluation criterion
	 *
	 * @return name GradientCriterion
	 */
	const char* get_name() const override { return "GradientCriterion"; }

private:
	/** evaluation direction */
	EEvaluationDirection m_direction;
};
}
#endif /* CGRADIENTCRITERION_H_ */
