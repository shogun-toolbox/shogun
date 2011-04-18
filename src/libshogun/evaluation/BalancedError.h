/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef BALANCEDERROR_H_
#define BALANCEDERROR_H_

#include "Evaluation.h"
#include "features/Labels.h"

namespace shogun
{

/** @brief The class BalancedError
 * used to compute balanced error of two-class classification
 *
 * Note this class is capable of evaluating only 2-class
 * labels.
 *
 */
class CBalancedError: public CEvaluation
{
public:
	/** constructor */
	CBalancedError() : CEvaluation() {};

	/** destructor */
	virtual ~CBalancedError() {};

	/** evaluate accuracy
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return balanced error
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** get name */
	virtual inline const char* get_name() const { return "Balanced error"; }

protected:

	/** get scores for TP, FP, TN, FN */
	void get_scores(CLabels* predicted, CLabels* ground_truth);

	// count of true positive labels
	float64_t m_TP;

	// count of false positive labels
	float64_t m_FP;

	// count of true negative labels
	float64_t m_TN;

	// count of false negative labels
	float64_t m_FN;
};

}


#endif /* BALANCEDERROR_H_ */
