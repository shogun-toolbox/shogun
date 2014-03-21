/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2011 Soeren Sonnenburg, Sergey Lisitsyn
 */

#ifndef MEANABSOLUTEERROR_H_
#define MEANABSOLUTEERROR_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CLabels;

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
class CMeanAbsoluteError: public CEvaluation
{
public:
	/** constructor */
	CMeanAbsoluteError() : CEvaluation() {};

	/** destructor */
	virtual ~CMeanAbsoluteError() {};

	/** evaluate mean absolute error
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return mean absolute error
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MINIMIZE;
	}

	/** get name */
	virtual const char* get_name() const { return "MeanAbsoluteError"; }
};

}

#endif /* MEANABSOLUTEERROR_H_ */
