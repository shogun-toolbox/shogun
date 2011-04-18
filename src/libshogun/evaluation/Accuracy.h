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

#include "BinaryClassEvaluation.h"
#include "features/Labels.h"

namespace shogun
{

/** @brief The class Accuracy
 * used to compute accuracy of classification.
 *
 * Formally, it is computed as
 * \f[
 * 		\frac{\mathsf{TP}+\mathsf{TN}}{\mathsf{N}},
 * \f]
 *
 * where TP is true positive rate, TN is true negative rate and
 * N is total number of labels.
 *
 * Note this class is capable of evaluating only 2-class
 * labels.
 *
 */
class CAccuracy: public CBinaryClassEvaluation
{
public:
	/** constructor */
	CAccuracy() : CBinaryClassEvaluation() {};

	/** destructor */
	virtual ~CAccuracy() {};

	/** evaluate accuracy
	 * @param predicted labels for evaluating
	 * @param ground_truth labels assumed to be correct
	 * @return accuracy
	 */
	virtual inline float64_t evaluate(CLabels* predicted, CLabels* ground_truth)
	{
		ASSERT(predicted->get_num_labels()==ground_truth->get_num_labels());
		get_scores(predicted, ground_truth);
		return (m_TP+m_TN)/(predicted->get_num_labels());
	}

	/** get name */
	virtual inline const char* get_name() const { return "2-class accuracy"; }
};

}

#endif /* ACCURACY_H_ */
