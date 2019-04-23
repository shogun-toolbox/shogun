/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Roman Votyakov, Evan Shelhamer, Yuyu Zhang, 
 *          Bjoern Esser
 */

#ifndef MEANABSOLUTEERROR_H_
#define MEANABSOLUTEERROR_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class Labels;

/** @brief Class MeanAbsoluteError
 * used to compute an error of regression model.
 *
 * Formally, for real labels \f$ L,R, |L|=|R|\f$ mean absolute
 * error (MAE) is estimated as
 *
 * \f[
 *		\frac{1}{|L|} \sum_{i=1}^{|L|} |L_i - R_i|
 * \f]
 *
 */
class MeanAbsoluteError: public Evaluation
{
public:
	/** constructor */
	MeanAbsoluteError() : Evaluation() {};

	/** destructor */
	virtual ~MeanAbsoluteError() {};

	/** evaluate mean absolute error
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return mean absolute error
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MINIMIZE;
	}

	/** get name */
	virtual const char* get_name() const { return "MeanAbsoluteError"; }
};

}

#endif /* MEANABSOLUTEERROR_H_ */
