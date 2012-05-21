/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef CONTINGENCYTABLEEVALUATION_H_
#define CONTINGENCYTABLEEVALUATION_H_

#include <shogun/evaluation/BinaryClassEvaluation.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

class CLabels;

/** type of measure */
enum EContingencyTableMeasureType
{
	ACCURACY = 0,
	ERROR_RATE = 10,
	BAL = 20,
	WRACC = 30,
	F1 = 40,
	CROSS_CORRELATION = 50,
	RECALL = 60,
	PRECISION = 70,
	SPECIFICITY = 80
};

/** @brief The class ContingencyTableEvaluation
 * a base class used to evaluate 2-class classification
 * with TP, FP, TN, FN rates.
 *
 * This class has implementations of the measures listed below:
 *
 * Accuracy (ACCURACY): \f$ \frac{TP+TN}{N} \f$
 *
 * Error rate (ERROR_RATE): \f$ \frac{FP+FN}{N} \f$
 *
 * Balanced error (BAL): \f$ \frac{1}{2} \left( \frac{FN}{FN+TP} + \frac{FP}{FP+TN} \right) \f$
 *
 * Weighted relative accuracy (WRACC): \f$ \frac{TP}{TP+FN} - \frac{FP}{FP+TN} \f$
 *
 * F1 score (F1): \f$ \frac{2\cdot FP}{2\cdot TP + FP + FN} \f$
 *
 * Cross correlation coefficient (CROSS_CORRELATION):
 * \f$ \frac{TP\cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \f$
 *
 * Recall (RECALL): \f$ \frac{TP}{TP+FN} \f$
 *
 * Precision (PRECISION): \f$ \frac{TP}{TP+FP} \f$
 *
 * Specificity (SPECIFICITY): \f$ \frac{TN}{TN+FP} \f$
 *
 * Note that objects of this class should be used only if
 * computing of many different measures is required. In other case,
 * using helper classes (CAccuracyMeasure, ...) could be more
 * convenient.
 *
 */
class CContingencyTableEvaluation: public CBinaryClassEvaluation
{

public:

	/** constructor */
	CContingencyTableEvaluation() :
		CBinaryClassEvaluation(), m_type(ACCURACY), m_computed(false) {};

	/** constructor
	 * @param type type of measure (e.g ACCURACY)
	 */
	CContingencyTableEvaluation(EContingencyTableMeasureType type) :
		CBinaryClassEvaluation(), m_type(type), m_computed(false)  {};

	/** destructor */
	virtual ~CContingencyTableEvaluation() {};

	/** evaluate labels
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(CLabels* predicted, CLabels* ground_truth);

	EEvaluationDirection get_evaluation_direction();

	/** get name */
	virtual inline const char* get_name() const
	{
		return "ContingencyTableEvaluation";
	}

	/** accuracy
	 * @return computed accuracy
	 */
	inline float64_t get_accuracy() const
	{
		if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first");

		return (m_TP+m_TN)/m_N;
	};

	/** error rate
	 * @return computed error rate
	 */
	inline float64_t get_error_rate() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return (m_FP + m_FN)/m_N;
	};

	/** Balanced error (BAL)
	 * @return computed BAL
	 */
	inline float64_t get_BAL() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return 0.5*(m_FN/(m_FN + m_TP) + m_FP/(m_FP + m_TN));
	};

	/** WRACC
	 * @return computed WRACC
	 */
	inline float64_t get_WRACC() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return m_TP/(m_FN + m_TP) - m_FP/(m_FP + m_TN);
	};

	/** F1
	 * @return computed F1 score
	 */
	inline float64_t get_F1() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return (2*m_TP)/(2*m_TP + m_FP + m_FN);
	};

	/** cross correlation
	 * @return computed cross correlation coefficient
	 */
	inline float64_t get_cross_correlation() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return (m_TP*m_TN-m_FP*m_FN)/CMath::sqrt((m_TP+m_FP)*(m_TP+m_FN)*(m_TN+m_FP)*(m_TN+m_FN));
	};

	/** recall
	 * @return computed recall
	 */
	inline float64_t get_recall() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return m_TP/(m_TP+m_FN);
	};

	/** precision
	 * @return computed precision
	 */
	inline float64_t get_precision() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return m_TP/(m_TP+m_FP);
	};

	/** specificity
	 * @return computed specificity
	 */
	inline float64_t get_specificity() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first");

		return m_TN/(m_TN+m_FP);
	};

protected:

	/** get scores for TP, FP, TN, FN */
	void compute_scores(CBinaryLabels* predicted, CBinaryLabels* ground_truth);

	/** type of measure to evaluate */
	EContingencyTableMeasureType m_type;

	/** indicator of contingencies being computed */
	bool m_computed;

	/** total number of labels */
	int32_t m_N;

	/** number of true positive examples */
	float64_t m_TP;

	/** number of false positive examples */
	float64_t m_FP;

	/** number of true negative examples */
	float64_t m_TN;

	/** number of false negative examples */
	float64_t m_FN;
};
}
#endif /* CONTINGENCYTABLEEVALUATION_H_ */
