/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ACCURACY_H_
#define ACCURACY_H_

#include "Evaluation.h"
#include "features/Labels.h"

namespace shogun
{

/** @brief The class Accuracy
 * used to compute accuracy of classification.
 *
 * Formally, for labels \f$L,R, |L|=|R|\f$ accuracy is estimated as
 *
 * \f[
 * 		\frac{\sum_{i=1}^{|L|} [\mathrm{sign} L_i=\mathrm{sign} R_i]}{|L|}
 * \f]
 *
 * Note this class is capable of evaluating only 2-class
 * labels.
 *
 */
class CAccuracy: public CEvaluation
{
public:
	/** constructor */
	CAccuracy() : CEvaluation() {};

	/** destructor */
	virtual ~CAccuracy() {};

	/** evaluate accuracy
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return accuracy
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** get name */
	virtual inline const char* get_name() const { return "Accuracy"; }
};

}

#endif /* ACCURACY_H_ */
