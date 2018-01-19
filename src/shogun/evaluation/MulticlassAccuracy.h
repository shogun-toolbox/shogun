/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Yuyu Zhang, Chiyuan Zhang, 
 *          Evan Shelhamer, Björn Esser, Roman Votyakov
 */

#ifndef MULTICLASSACCURACY_H_
#define MULTICLASSACCURACY_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

class CLabels;

/** @brief The class MulticlassAccuracy
 * used to compute accuracy of multiclass classification.
 *
 * Formally, for multiclass labels \f$L,R, |L|=|R|\f$ multiclass
 * accuracy is estimated as
 *
 * \f[
 *		\frac{\sum_{i=1}^{|L|} [L_i=R_i]}{|L|}
 * \f]
 *
 *
 */
class CMulticlassAccuracy: public CEvaluation
{
public:
	/** constructor */
	CMulticlassAccuracy() :
		CEvaluation(), m_ignore_rejects(false), m_rejects_num(0) {};

	/** constructor */
	CMulticlassAccuracy(bool ignore_rejects) :
		CEvaluation(), m_ignore_rejects(ignore_rejects), m_rejects_num(0) {};

	/** destructor */
	virtual ~CMulticlassAccuracy() {};

	/** evaluate accuracy
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 * @return accuracy
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	/** constructs confusion matrix for multiclass classification
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 * @return confusion matrix
	 */
	static SGMatrix<int32_t> get_confusion_matrix(CLabels* predicted, CLabels* ground_truth);

	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** get rejects num */
	int32_t get_rejects_num() const
	{
		return m_rejects_num;
	}

	/** get name */
	virtual const char* get_name() const { return "MulticlassAccuracy"; }

protected:

	/** ignore rejects */
	bool m_ignore_rejects;

	/** rejects num */
	int32_t m_rejects_num;
};

}

#endif /* MULTICLASSACCURACY_H_ */
