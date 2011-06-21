/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef EVALUATION_H_
#define EVALUATION_H_

#include "features/Labels.h"
#include "base/SGObject.h"

namespace shogun
{

class CLabels;

/** enum which is used to define whether an evaluation criterium has to be
 * minimised or maximised */
enum EEvaluationDirection
{
	ED_MINIMIZE, ED_MAXIMIZE
};

/** @brief The class Evaluation
 * a main class for other classes
 * used to evaluate labels, e.g. accuracy of classification or
 * mean squared error of regression.
 *
 * This class provides only interface for evaluation with
 * no proposed implementation.
 *
 */
class CEvaluation: public CSGObject
{
public:
	/** constructor */
	CEvaluation() : CSGObject() {};

	/** destructor */
	virtual ~CEvaluation() {};

	/** evaluate labels
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth) = 0;

	/** @return whether criterium has to be maximised or minimised */
	virtual EEvaluationDirection get_evaluation_direction()=0;
};

}

#endif /* EVALUATION_H_ */
