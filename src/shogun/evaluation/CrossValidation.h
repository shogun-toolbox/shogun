/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __CROSSVALIDATION_H_
#define __CROSSVALIDATION_H_

#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/evaluation/MachineEvaluation.h>

namespace shogun
{

class CMachineEvaluation;
class CCrossValidationOutput;
class CList;

/** @brief type to encapsulate the results of an evaluation run.
 * May contain confidence interval (if conf_int_alpha!=0).
 * m_conf_int_alpha is the probability for an error, i.e. the value does not lie
 * in the confidence interval.
 */
class CCrossValidationResult : public CEvaluationResult
{
	public:
		CCrossValidationResult()
		{
			SG_ADD(&mean, "mean", "Mean of results", MS_NOT_AVAILABLE);
			SG_ADD(&has_conf_int, "has_conf_int", "Has confidence intervals?",
					MS_NOT_AVAILABLE);
			SG_ADD(&conf_int_low, "conf_int_low", "Lower confidence bound",
					MS_NOT_AVAILABLE);
			SG_ADD(&conf_int_up, "conf_int_up", "Upper confidence bound",
						MS_NOT_AVAILABLE);

			SG_ADD(&conf_int_alpha, "conf_int_alpha",
					"Alpha of confidence interval", MS_NOT_AVAILABLE);

			mean = 0;
			has_conf_int = 0;
			conf_int_low = 0;
			conf_int_up = 0;
			conf_int_alpha = 0;
		}

		/** return what type of result we are.
		 *
		 * @return result type
		 */
		virtual EEvaluationResultType get_result_type() const
		{
			return CROSSVALIDATION_RESULT;
		}

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "CrossValidationResult"; }

		/** helper method used to specialize a base class instance
		 *
		 * @param eval_result its dynamic type must be CCrossValidationResult
		 */
		static CCrossValidationResult* obtain_from_generic(
				CEvaluationResult* eval_result)
		{
			if (!eval_result)
				return NULL;

			REQUIRE(eval_result->get_result_type()==CROSSVALIDATION_RESULT,
					"CrossValidationResult::obtain_from_generic(): argument is"
					"of wrong type!\n");

			SG_REF(eval_result);
			return (CCrossValidationResult*) eval_result;
		}

		/** print result */
		virtual void print_result()
		{
			if (has_conf_int)
			{
				SG_SPRINT("[%f,%f] with alpha=%f, mean=%f\n", conf_int_low,
						conf_int_up, conf_int_alpha, mean);
			}
			else
				SG_SPRINT("%f\n", mean)
		}

	public:
		/** mean */
		float64_t mean;
		/** has conf int */
		bool has_conf_int;
		/** conf int low */
		float64_t conf_int_low;
		/** conf int up */
		float64_t conf_int_up;
		/** conf int alpha */
		float64_t conf_int_alpha;

};

/** @brief base class for cross-validation evaluation.
 * Given a learning machine, a splitting strategy, an evaluation criterium,
 * features and correspnding labels, this provides an interface for
 * cross-validation. Results may be retrieved using the evaluate method. A
 * number of repetitions may be specified for obtaining more accurate results.
 * The arithmetic mean of different runs is returned along with confidence
 * intervals, if a p-value is specified.
 * Default number of runs is one, confidence interval combutation is disabled.
 *
 * This class calculates an evaluation criterium of every fold and then
 * calculates the arithmetic mean of all folds. This is for example suitable
 * for the AUC or for Accuracy. However, for example F1-measure may not be
 * merged this way (result will be biased). To solve this, different sub-classes
 * may average results of each cross validation fold differently by overwriting
 * the evaluate_one_run method.
 *
 * See [Forman, G. and Scholz, M. (2009). Apples-to-apples in cross-validation
 * studies: Pitfalls in classifier performance measurement. Technical report,
 * HP Laboratories.] for details on this subject.
 *
 * Cross validation tries to lock underlying machines if that is possible to
 * speed up computations. Can be turned off by the set_autolock()  method.
 * Locking in general may speed up things (eg for kernel machines the kernel
 * matrix is precomputed), however, it is not always supported.
 */
class CCrossValidation: public CMachineEvaluation
{
public:
	/** constructor */
	CCrossValidation();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CCrossValidation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock=true);

	/** constructor, for use with custom kernels (no features)
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion evaluation criterion to use
	 * @param autolock autolock
	 */
	CCrossValidation(CMachine* machine, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock=true);

	/** destructor */
	virtual ~CCrossValidation();

	/** setter for the number of runs to use for evaluation */
	void set_num_runs(int32_t num_runs);

	/** setter for the number of runs to use for evaluation */
	void set_conf_int_alpha(float64_t m_conf_int_alpha);

	/** evaluate */
	virtual CEvaluationResult* evaluate();

	/** appends given cross validation output instance
	 * to the list of listeners
	 *
	 * @param cross_validation_output given cross validation output
	 */
	void add_cross_validation_output(
			CCrossValidationOutput* cross_validation_output);

	/** @return name of the SGSerializable */
	virtual const char* get_name() const
	{
		return "CrossValidation";
	}

private:
	void init();

protected:
	/** Evaluates one single cross-validation run.
	 * Current implementation evaluates each fold separately and then calculates
	 * arithmetic mean. Suitable for accuracy and AUC for example. NOT for
	 * F1-measure. Has to be overridden by sub-classes if results have to be
	 * merged differently
	 *
	 * @return evaluation result of one cross-validation run
	 */
	virtual float64_t evaluate_one_run();

	/** number of evaluation runs for one fold */
	int32_t m_num_runs;
	/** confidence interval alpha parameter */
	float64_t m_conf_int_alpha;

	/** xval output listeners */
	CList* m_xval_outputs;
};

}

#endif /* __CROSSVALIDATION_H_ */
