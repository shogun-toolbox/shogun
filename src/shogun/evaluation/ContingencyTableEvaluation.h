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

#include <evaluation/BinaryClassEvaluation.h>
#include <labels/Labels.h>
#include <mathematics/Math.h>
#include <io/SGIO.h>

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
	SPECIFICITY = 80,
	CUSTOM = 999
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

	virtual EEvaluationDirection get_evaluation_direction() const;

	/** get name */
	virtual const char* get_name() const
	{
		return "ContingencyTableEvaluation";
	}

	/** accuracy
	 * @return computed accuracy
	 */
	inline float64_t get_accuracy() const
	{
		if (!m_computed)
			SG_ERROR("Uninitialized, please call evaluate first")

		return (m_TP+m_TN)/m_N;
	};

	/** error rate
	 * @return computed error rate
	 */
	inline float64_t get_error_rate() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return (m_FP + m_FN)/m_N;
	};

	/** Balanced error (BAL)
	 * @return computed BAL
	 */
	inline float64_t get_BAL() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return 0.5*(m_FN/(m_FN + m_TP) + m_FP/(m_FP + m_TN));
	};

	/** WRACC
	 * @return computed WRACC
	 */
	inline float64_t get_WRACC() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return m_TP/(m_FN + m_TP) - m_FP/(m_FP + m_TN);
	};

	/** F1
	 * @return computed F1 score
	 */
	inline float64_t get_F1() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return (2*m_TP)/(2*m_TP + m_FP + m_FN);
	};

	/** cross correlation
	 * @return computed cross correlation coefficient
	 */
	inline float64_t get_cross_correlation() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return (m_TP*m_TN-m_FP*m_FN)/CMath::sqrt((m_TP+m_FP)*(m_TP+m_FN)*(m_TN+m_FP)*(m_TN+m_FN));
	};

	/** recall
	 * @return computed recall
	 */
	inline float64_t get_recall() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return m_TP/(m_TP+m_FN);
	};

	/** precision
	 * @return computed precision
	 */
	inline float64_t get_precision() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return m_TP/(m_TP+m_FP);
	};

	/** specificity
	 * @return computed specificity
	 */
	inline float64_t get_specificity() const
	{
		if (!m_computed)
				SG_ERROR("Uninitialized, please call evaluate first")

		return m_TN/(m_TN+m_FP);
	};

	/** Returns number of True Positives
	 * @return number of true positives
	 */
	float64_t get_TP() const
	{
		return m_TP;
	}
	/** Returns number of False Positives
	 * @return number of false positives
	 */
	float64_t get_FP() const
	{
		return m_FP;
	}
	/** Returns number of True Negatives
	 * @return number of true negatives
	 */
	float64_t get_TN() const
	{
		return m_TN;
	}
	/** Returns number of True Negatives
	 * @return number of false negatives
	 */
	float64_t get_FN() const
	{
		return m_FN;
	}

	/** Computes custom score, not implemented
	 * @return custom score value
	 */
	virtual float64_t get_custom_score()
	{
		SG_NOTIMPLEMENTED
		return 0.0;
	}

	/** Returns custom direction, not implemented
	 * @return direction of custom score
	 */
	virtual EEvaluationDirection get_custom_direction() const
	{
		SG_NOTIMPLEMENTED
		return ED_MAXIMIZE;
	}

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

/** @brief class AccuracyMeasure
 * used to measure accuracy of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CAccuracyMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CAccuracyMeasure() : CContingencyTableEvaluation(ACCURACY) {};
	/* virtual destructor */
	virtual ~CAccuracyMeasure() {};
	/* name */
	virtual const char* get_name() const { return "AccuracyMeasure"; };
};

/** @brief class ErrorRateMeasure
 * used to measure error rate of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CErrorRateMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CErrorRateMeasure() : CContingencyTableEvaluation(ERROR_RATE) {};
	/* virtual destructor */
	virtual ~CErrorRateMeasure() {};
	/* name */
	virtual const char* get_name() const { return "ErrorRateMeasure"; };
};

/** @brief class BALMeasure
 * used to measure balanced error of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CBALMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CBALMeasure() : CContingencyTableEvaluation(BAL) {};
	/* virtual destructor */
	virtual ~CBALMeasure() {};
	/* name */
	virtual const char* get_name() const { return "BALMeasure"; };
};

/** @brief class WRACCMeasure
 * used to measure weighted relative accuracy of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CWRACCMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CWRACCMeasure() : CContingencyTableEvaluation(WRACC) {};
	/* virtual destructor */
	virtual ~CWRACCMeasure() {};
	/* name */
	virtual const char* get_name() const { return "WRACCMeasure"; };
};

/** @brief class F1Measure
 * used to measure F1 score of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CF1Measure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CF1Measure() : CContingencyTableEvaluation(F1) {};
	/* virtual destructor */
	virtual ~CF1Measure() {};
	/* name */
	virtual const char* get_name() const { return "F1Measure"; };
};

/** @brief class CrossCorrelationMeasure
 * used to measure cross correlation coefficient of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CCrossCorrelationMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CCrossCorrelationMeasure() : CContingencyTableEvaluation(CROSS_CORRELATION) {};
	/* virtual destructor */
	virtual ~CCrossCorrelationMeasure() {};
	/* name */
	virtual const char* get_name() const { return "CrossCorrelationMeasure"; };
};

/** @brief class RecallMeasure
 * used to measure recall of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CRecallMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CRecallMeasure() : CContingencyTableEvaluation(RECALL) {};
	/* virtual destructor */
	virtual ~CRecallMeasure() {};
	/* name */
	virtual const char* get_name() const { return "RecallMeasure"; };
};

/** @brief class PrecisionMeasure
 * used to measure precision of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CPrecisionMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CPrecisionMeasure() : CContingencyTableEvaluation(PRECISION) {};
	/* virtual destructor */
	virtual ~CPrecisionMeasure() {};
	/* name */
	virtual const char* get_name() const { return "PrecisionMeasure"; };
};

/** @brief class SpecificityMeasure
 * used to measure specificity of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of CContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CSpecificityMeasure: public CContingencyTableEvaluation
{
public:
	/* constructor */
	CSpecificityMeasure() : CContingencyTableEvaluation(SPECIFICITY) {};
	/* virtual destructor */
	virtual ~CSpecificityMeasure() {};
	/* name */
	virtual const char* get_name() const { return "SpecificityMeasure"; };
};
}
#endif /* CONTINGENCYTABLEEVALUATION_H_ */
