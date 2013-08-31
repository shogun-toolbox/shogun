/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __MEANSQUAREDLOGERROR__
#define __MEANSQUAREDLOGERROR__

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CLabels;

/** @brief Class CMeanSquaredLogError
 * used to compute an error of regression model.
 *
 * Formally, for real labels \f$ L,R, |L|=|R|, L_i, R_i > -1\f$ mean squared
 * log error is estimated as
 *
 * \f[
 * 		\sqrt{\frac{1}{|L|} \sum_{i=1}^{|L|} (\log{L_i+1} - \log{R_i+1})^2}
 * \f]
 *
 */
class CMeanSquaredLogError: public CEvaluation
{
public:
	/** constructor */
	CMeanSquaredLogError() : CEvaluation() {};

	/** destructor */
	virtual ~CMeanSquaredLogError() {};

	/** evaluate mean squared log error
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
	virtual const char* get_name() const { return "MeanSquaredLogError"; }
};

}

#endif /* __MEANSQUAREDLOGERROR__ */
