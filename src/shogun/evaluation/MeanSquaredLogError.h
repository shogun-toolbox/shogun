/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Roman Votyakov, Yuyu Zhang, 
 *          Bjoern Esser
 */

#ifndef __MEANSQUAREDLOGERROR__
#define __MEANSQUAREDLOGERROR__

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class Labels;

/** @brief Class CMeanSquaredLogError
 * used to compute an error of regression model.
 *
 * Formally, for real labels \f$ L,R, |L|=|R|, L_i, R_i > -1\f$ mean squared
 * log error is estimated as
 *
 * \f[
 *		\sqrt{\frac{1}{|L|} \sum_{i=1}^{|L|} (\log{L_i+1} - \log{R_i+1})^2}
 * \f]
 *
 */
class MeanSquaredLogError: public Evaluation
{
public:
	/** constructor */
	MeanSquaredLogError() : Evaluation() {};

	/** destructor */
	virtual ~MeanSquaredLogError() {};

	/** evaluate mean squared log error
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return mean squared error
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MINIMIZE;
	}

	/** get name */
	virtual const char* get_name() const { return "MeanSquaredLogError"; }
};

}

#endif /* __MEANSQUAREDLOGERROR__ */
