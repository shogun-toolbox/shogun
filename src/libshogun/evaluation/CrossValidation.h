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
class CSplittingStrategy;
class CEvaluation;

/** @brief abstract base class for cross-validation evaluation.
 * Given a learning machine, a splitting strategy and an evaluation criterium,
 * this provides an interface for cross-validation. Results may be retrieved
 * using the evaluate method. A number of repetitions may be specified for
 * obtaining more accurate results. The arithmetic mean of different runs is
 * returned along with confidence intervals for a given p-value.
 *
 * Different implementations may average results of each cross validation fold
 * differently. For example, AUC has to be averaged differently than the
 * f1-measure. For every evaluation criterium, a new class has to implemented,
 * so that the fact that the method of averaging is important may not be ignored.
 *
 * Sub-classes HAVE to override the evaluate_one_run method, which is empty here
 * and which is called by the evaulate method.
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
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterium evaluation criterium to use
	 */
	CCrossValidation(CMachine* machine, CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterium);

	/** destructor */
	virtual ~CCrossValidation();

	/** @return in which direction is the best evaluation value? */
	inline EEvaluationDirection get_evaluation_direction()
	{
		return m_evaluation_criterium->get_evaluation_direction();
	}

	/** abstract method for evaluation.
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
	virtual float64_t evaluate(int32_t num_runs=1, float64_t conf_int_p=0,
			float64_t* conf_int_low=NULL, float64_t* conf_int_up=NULL);

	/** @return name of the SGSerializable */
	inline virtual const char* get_name() const
	{
		return "CrossValidation";
	}

protected:
	/** evaulates one single cross-validation run.
	 * Has to be overriden by sub-classes. Be careful when averaging results of
	 * different folds (see class description)
	 */
	virtual float64_t evaluate_one_run() { return 0; }

private:
	CMachine* m_machine;
	CSplittingStrategy* m_splitting_strategy;
	CEvaluation* m_evaluation_criterium;
};

}

#endif /* __CROSSVALIDATION_H_ */
