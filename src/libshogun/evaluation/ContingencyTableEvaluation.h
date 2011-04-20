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

#include "evaluation/BinaryClassEvaluation.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/io.h"

namespace shogun
{

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
	SPECIFITY = 80
};

/** @brief The class ContingencyTableEvaluation
 * a base class used to evaluate 2-class classification
 * with TP, FP, TN, FN rates.
 *
 * This class have implementations of measures listed below:
 *
 * Accuracy (ACCURACY): \f$ \frac{TP+TN}{N} \f$
 *
 * Error rate (ERROR_RATE): \f$ \frac{FP+FN}{N} \f$
 *
 * Balanced error (BAL): \f$ \frac{1}{2} \left( \frac{FN}{FN+TP} + \frac{FP}{FP+TN} \right) \f$
 *
 * Weighted relative accuracy (WRACC): \f$ \frac{TP}{TP+FN} - \frac{FP}{FP+TN} \f$
 *
 * F1 score (F!): \f$ \frac{2\cdot FP}{2\cdot TP + FP + FN} \f$
 *
 * Cross correlation (CROSS_CORRELATION):
 * \f$ \frac{TP\cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \f$
 *
 * Recall (RECALL): \f$ \frac{TP}{TP+FN} \f$
 *
 * Precision (PRECISION): \f$ \frac{TP}{TP+FP} \f$
 *
 * Specifity (SPECIFITY): \f$ \frac{TN}{TN+FP} \f$
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

	/** get name */
	virtual inline const char* get_name() const { return "ContingencyTableEvaluation"; }

	/* accuracy */
	inline float64_t get_accuracy() const
	{
		if (!m_computed)
			SG_ERROR("Uninitialized");
		return (m_TP+m_TN)/m_N;
	};

	/* error rate */
	inline float64_t get_error_rate() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return (m_FP + m_FN)/m_N;
	};

	/* BAL */
	inline float64_t get_BAL() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return 0.5*(m_FN/(m_FN + m_TP) + m_FP/(m_FP + m_TN));
	};

	/* WRACC */
	inline float64_t get_WRACC() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return m_TP/(m_FN + m_TP) - m_FP/(m_FP + m_TN);
	};

	/* F1 */
	inline float64_t get_F1() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return (2*m_TP)/(2*m_TP + m_FP + m_FN);
	};

	/* cross correlation */
	inline float64_t get_cross_correlation() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return (m_TP*m_TN-m_FP*m_FN)/CMath::sqrt((m_TP+m_FP)*(m_TP+m_FN)*(m_TN+m_FP)*(m_TN+m_FN));
	};

	/* recall */
	inline float64_t get_recall() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return m_TP/(m_TP+m_FN);
	};

	/* precision */
	inline float64_t get_precision() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return m_TP/(m_TP+m_FP);
	};

	/* specifity */
	inline float64_t get_specifity() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized");
		return m_TN/(m_TN+m_FP);
	};

protected:

	/** get scores for TP, FP, TN, FN */
	void compute_scores(CLabels* predicted, CLabels* ground_truth);

	// type of measure
	EContingencyTableMeasureType m_type;

	// indicator of scores being computed already
	bool m_computed;

	// number of labels
	int32_t m_N;

	// count of true positive labels
	float64_t m_TP;

	// count of false positive labels
	float64_t m_FP;

	// count of true negative labels
	float64_t m_TN;

	// count of false negative labels
	float64_t m_FN;
};

/** @brief class Accuracy
 * used to measure accuracy of 2-class classifier
 */
class CAccuracy: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CAccuracy() : CContingencyTableEvaluation(ACCURACY) {};
	/* virtual destructor */
	virtual ~CAccuracy() {};
	/* name */
	virtual inline const char* get_name() const { return "Accuracy"; };
};

/** @brief class ErrorRate
 * used to measure error rate of 2-class classifier
 */
class CErrorRate: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CErrorRate() : CContingencyTableEvaluation(ERROR_RATE) {};
	/* virtual destructor */
	virtual ~CErrorRate() {};
	/* name */
	virtual inline const char* get_name() const { return "ErrorRate"; };
};

/** @brief class BAL
 * used to measure balanced error of 2-class classifier
 */
class CBAL: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CBAL() : CContingencyTableEvaluation(BAL) {};
	/* virtual destructor */
	virtual ~CBAL() {};
	/* name */
	virtual inline const char* get_name() const { return "BAL"; };
};

}


#endif /* CONTINGENCYTABLEEVALUATION_H_ */
