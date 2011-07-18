/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __CROSSVALIDATION_H_
#define __CROSSVALIDATION_H_

#include <shogun/base/SGObject.h>
#include <shogun/evaluation/Evaluation.h>

namespace shogun
{

class CMachine;
class CFeatures;
class CLabels;
class CSplittingStrategy;
class CEvaluation;

typedef struct
{
	float64_t value;
	bool has_conf_int;
	float64_t conf_int_low;
	float64_t conf_int_up;
	float64_t conf_int_p;

	void print_result()
	{
		if (has_conf_int)
		{
			SG_SPRINT("[%f,%f] with p=%f, value=%f\n", conf_int_low, conf_int_up,
					conf_int_p);
		}
		else
			SG_SPRINT("%f\n", value);
	}
} CrossValidationResult;

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
 */
class CCrossValidation: public CSGObject
{
public:
	/** constructor */
	CCrossValidation();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterium evaluation criterium to use
	 */
	CCrossValidation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterium);

	/** destructor */
	virtual ~CCrossValidation();

	/** @return in which direction is the best evaluation value? */
	inline EEvaluationDirection get_evaluation_direction()
	{
		return m_evaluation_criterium->get_evaluation_direction();
	}

	/** method for evaluation. Performs cross-validation.
	 * Is repeated m_num_runs. If this number is larger than one, a confidence
	 * interval is calculated if m_conf_int_p is (0<p<1).
	 * By default m_num_runs=1 and m_conf_int_p=0
	 *
	 * @return result of evaluation
	 */
	CrossValidationResult evaluate();

	/** @return parameter instance of underlying learning machine */
	Parameter* get_machine_parameters() const;

	/** setter for the number of runs to use for evaluation */
	void set_num_runs(int32_t num_runs);

	/** setter for the number of runs to use for evaluation */
	void set_conf_int_p(float64_t conf_int_p);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
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

private:
	int32_t m_num_runs;
	float64_t m_conf_int_p;

	CMachine* m_machine;
	CFeatures* m_features;
	CLabels* m_labels;
	CSplittingStrategy* m_splitting_strategy;
	CEvaluation* m_evaluation_criterium;
};

}

#endif /* __CROSSVALIDATION_H_ */
