/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef BINARYCLASSEVALUATION_H_
#define BINARYCLASSEVALUATION_H_

#include "evaluation/Evaluation.h"
#include "features/Labels.h"

namespace shogun
{

/** @brief The class TwoClassEvaluation
 * a base class used to evaluate 2-class classification
 *
 */
class CBinaryClassEvaluation: public CEvaluation
{

public:

	/** constructor */
	CBinaryClassEvaluation() : CEvaluation() {};

	/** destructor */
	virtual ~CBinaryClassEvaluation() {};

	/** evaluate labels
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth) = 0;
};

}


#endif /* BINARYCLASSEVALUATION_H_ */
