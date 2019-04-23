/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Roman Votyakov, Yuyu Zhang
 */

#ifndef CONTINGENCYTABLEEVALUATION_H_
#define CONTINGENCYTABLEEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/BinaryClassEvaluation.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

class Labels;

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
class ContingencyTableEvaluation: public BinaryClassEvaluation
{

public:

	/** constructor */
	ContingencyTableEvaluation() :
		BinaryClassEvaluation(), m_type(ACCURACY), m_computed(false) {};

	/** constructor
	 * @param type type of measure (e.g ACCURACY)
	 */
	ContingencyTableEvaluation(EContingencyTableMeasureType type) :
		BinaryClassEvaluation(), m_type(type), m_computed(false)  {};

	/** destructor */
	virtual ~ContingencyTableEvaluation() {};

	/** evaluate labels
	 * @param predicted labels
	 * @param ground_truth labels assumed to be correct
	 * @return evaluation result
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth);

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

		return (m_TP * m_TN - m_FP * m_FN) / std::sqrt(
			                                     (m_TP + m_FP) * (m_TP + m_FN) *
			                                     (m_TN + m_FP) * (m_TN + m_FN));
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
	void compute_scores(std::shared_ptr<BinaryLabels> predicted, std::shared_ptr<BinaryLabels> ground_truth);

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
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class AccuracyMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	AccuracyMeasure() : ContingencyTableEvaluation(ACCURACY) {};
	/* virtual destructor */
	virtual ~AccuracyMeasure() {};
	/* name */
	virtual const char* get_name() const { return "AccuracyMeasure"; };
};

/** @brief class ErrorRateMeasure
 * used to measure error rate of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class ErrorRateMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	ErrorRateMeasure() : ContingencyTableEvaluation(ERROR_RATE) {};
	/* virtual destructor */
	virtual ~ErrorRateMeasure() {};
	/* name */
	virtual const char* get_name() const { return "ErrorRateMeasure"; };
};

/** @brief class BALMeasure
 * used to measure balanced error of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class BALMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	BALMeasure() : ContingencyTableEvaluation(BAL) {};
	/* virtual destructor */
	virtual ~BALMeasure() {};
	/* name */
	virtual const char* get_name() const { return "BALMeasure"; };
};

/** @brief class WRACCMeasure
 * used to measure weighted relative accuracy of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class WRACCMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	WRACCMeasure() : ContingencyTableEvaluation(WRACC) {};
	/* virtual destructor */
	virtual ~WRACCMeasure() {};
	/* name */
	virtual const char* get_name() const { return "WRACCMeasure"; };
};

/** @brief class F1Measure
 * used to measure F1 score of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class F1Measure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	F1Measure() : ContingencyTableEvaluation(F1) {};
	/* virtual destructor */
	virtual ~F1Measure() {};
	/* name */
	virtual const char* get_name() const { return "F1Measure"; };
};

/** @brief class CrossCorrelationMeasure
 * used to measure cross correlation coefficient of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class CrossCorrelationMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	CrossCorrelationMeasure() : ContingencyTableEvaluation(CROSS_CORRELATION) {};
	/* virtual destructor */
	virtual ~CrossCorrelationMeasure() {};
	/* name */
	virtual const char* get_name() const { return "CrossCorrelationMeasure"; };
};

/** @brief class RecallMeasure
 * used to measure recall of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class RecallMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	RecallMeasure() : ContingencyTableEvaluation(RECALL) {};
	/* virtual destructor */
	virtual ~RecallMeasure() {};
	/* name */
	virtual const char* get_name() const { return "RecallMeasure"; };
};

/** @brief class PrecisionMeasure
 * used to measure precision of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class PrecisionMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	PrecisionMeasure() : ContingencyTableEvaluation(PRECISION) {};
	/* virtual destructor */
	virtual ~PrecisionMeasure() {};
	/* name */
	virtual const char* get_name() const { return "PrecisionMeasure"; };
};

/** @brief class SpecificityMeasure
 * used to measure specificity of 2-class classifier.
 *
 * This class is also capable of measuring
 * any other rate using get_[measure name] methods
 * of ContingencyTableEvaluation class.
 *
 * Note that evaluate() should be called first.
 */
class SpecificityMeasure: public ContingencyTableEvaluation
{
public:
	/* constructor */
	SpecificityMeasure() : ContingencyTableEvaluation(SPECIFICITY) {};
	/* virtual destructor */
	virtual ~SpecificityMeasure() {};
	/* name */
	virtual const char* get_name() const { return "SpecificityMeasure"; };
};
}
#endif /* CONTINGENCYTABLEEVALUATION_H_ */
