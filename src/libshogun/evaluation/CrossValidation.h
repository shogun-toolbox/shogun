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

#include "base/SGObject.h"
#include "evaluation/Evaluation.h"

namespace shogun
{

class CMachine;
class CFeatures;
class CLabels;
class CSplittingStrategy;
class CEvaluation;

/** @brief base class for cross-validation evaluation.
 * Given a learning machine, a splitting strategy, an evaluation criterium,
 * features and crrespnding labels, this provides an interface for
 * cross-validation. Results may be retrieved using the evaluate method. A
 * number of repetitions may be specified for obtaining more accurate results.
 * The arithmetic mean of different runs is returned along with confidence
 * intervals for a given p-value.
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

	/** method for evaluation.
	 * A number of runs may be specified for repetition of procedure. If this
	 * number is larger than one, a confidence interval may be calculated if
	 * provided pointers are non-NULL. A p-value may be specified for this.
	 *
	 * @param num_runs number of repetitions. if >1 and the other parameters
	 * are non-NULL, a confidence interval is calculated
	 * @param conf_int_p probability that real value lies in confidence interval
	 * @param conf_int_low lower bound of confidence interval is written here
	 * @param conf_int_up upper bound of confidence interval is written here
	 *
	 * @return arithmetic mean of cross-validation runs
	 */
	float64_t evaluate(int32_t num_runs=1, float64_t conf_int_p=0,
			float64_t* conf_int_low=NULL, float64_t* conf_int_up=NULL);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "CrossValidation";
	}

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
	CMachine* m_machine;
	CFeatures* m_features;
	CLabels* m_labels;
	CSplittingStrategy* m_splitting_strategy;
	CEvaluation* m_evaluation_criterium;
};

}

#endif /* __CROSSVALIDATION_H_ */
