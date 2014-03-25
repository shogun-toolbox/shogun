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

#include <shogun/lib/config.h>
#include <shogun/labels/Labels.h>
#include <shogun/base/SGObject.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
class CLabels;

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
class CEvaluation : public CSGObject
{
public:
	/** default constructor */
	CEvaluation() : CSGObject() { };

	/** destructor */
	virtual ~CEvaluation() { };

	/** evaluate labels
	 *
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth)=0;

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
