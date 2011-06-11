/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef MULTICLASSACCURACY_H_
#define MULTICLASSACCURACY_H_

#include "evaluation/Evaluation.h"
#include "features/Labels.h"

namespace shogun
{

class CLabels;

/** @brief The class MulticlassAccuracy
 * used to compute accuracy of multiclass classification.
 *
 * Formally, for labels \f$L,R, |L|=|R|\f$ accuracy is estimated as
 *
 * \f[
 * 		\frac{\sum_{i=1}^{|L|} [L_i=R_i]}{|L|}
 * \f]
 *
 *
 */
class CMulticlassAccuracy: public CEvaluation
{
public:
	/** constructor */
	CMulticlassAccuracy() : CEvaluation() {};

	/** destructor */
	virtual ~CMulticlassAccuracy() {};

	/** evaluate accuracy
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return accuracy
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	inline EEvaluationDirection get_evaluation_direction()
	{
		return ED_MINIMISE;
	}

	/** get name */
	virtual inline const char* get_name() const { return "MulticlassAccuracy"; }
};

}

#endif /* MULTICLASSACCURACY_H_ */
