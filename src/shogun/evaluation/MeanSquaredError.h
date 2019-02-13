/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Evan Shelhamer, Bjoern Esser, 
 *          Sergey Lisitsyn, Roman Votyakov
 */

#ifndef MEANSQUAREDERROR_H_
#define MEANSQUAREDERROR_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CLabels;

/** @brief Class MeanSquaredError
 * used to compute an error of regression model.
 *
 * Formally, for real labels \f$ L,R, |L|=|R|\f$ mean squared
 * error (MSE) is estimated as
 *
 * \f[
 *		\frac{1}{|L|} \sum_{i=1}^{|L|} (L_i - R_i)^2
 * \f]
 *
 */
class SHOGUN_EXPORT CMeanSquaredError: public CEvaluation
{
public:
	/** constructor */
	CMeanSquaredError() : CEvaluation() {};

	/** destructor */
	virtual ~CMeanSquaredError() {};

	/** evaluate mean squared error
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return mean squared error
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MINIMIZE;
	}

	/** get name */
	virtual const char* get_name() const { return "MeanSquaredError"; }
};

}

#endif /* MEANSQUAREDERROR_H_ */
