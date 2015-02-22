/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Some code adapted from CrossValidation class by
 * Heiko Strathmann
 */

#ifndef CMACHINEEVALUATION_H_
#define CMACHINEEVALUATION_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/evaluation/MachineEvaluation.h>

namespace shogun
{

class CMachine;
class CFeatures;
class CLabels;
class CSplittingStrategy;
class CEvaluation;

/** @brief Machine Evaluation is an abstract class
 * that evaluates a machine according to some criterion.
 *
 */
class CMachineEvaluation: public CSGObject
{

public:

	CMachineEvaluation();

	/** constructor
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criteria list of evaluation criteria to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
	CMachineEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CDynamicObjectArray* evaluation_criteria,
			bool autolock = true);

	/** constructor with a single evaluation criterion.
         * Kept for convenience
	 * @param machine learning machine to use
	 * @param features features to use for cross-validation
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion takes a single evaluation criterion to use
	 * @param autolock whether machine should be auto-locked before evaluation
	 */
        CMachineEvaluation(CMachine* machine, CFeatures* features, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock = true);

	/** constructor, for use with custom kernels (no features)
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criteria list of evaluation criteria to use
	 * @param autolock autolock
	 */
	CMachineEvaluation(CMachine* machine, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CDynamicObjectArray* evaluation_criteria,
			bool autolock = true);

	/** constructor, for use with custom kernels (no features)
     * with a single evaluation criterion. Kept for convenience
	 * @param machine learning machine to use
	 * @param labels labels that correspond to the features
	 * @param splitting_strategy splitting strategy to use
	 * @param evaluation_criterion takes a single evaluation criterion to use
	 * @param autolock autolock
	 */
        CMachineEvaluation(CMachine* machine, CLabels* labels,
			CSplittingStrategy* splitting_strategy,
			CEvaluation* evaluation_criterion, bool autolock = true);

	virtual ~CMachineEvaluation();

	/** @return in which direction is the best evaluation value? */
	EEvaluationDirection get_evaluation_direction();

	/** @return in which direction is the best evaluation value
	 *  for the corresponding evaluation criterion
	 */
	CDynamicArray<EEvaluationDirection> get_evaluation_directions();

	/** method for evaluation. Performs cross-validation.
	 * Is repeated m_num_runs. If this number is larger than one, a confidence
	 * interval is calculated if m_conf_int_alpha is (0<p<1).
	 * By default m_num_runs=1 and m_conf_int_alpha=0
	 *
	 * @return result of evaluation
	 */
	virtual CEvaluationResult* evaluate() = 0;

	/** @return underlying learning machine */
	CMachine* get_machine() const;

	/** setter for the autolock property. If true, machine will tried to be
	 * locked before evaluation */
	void set_autolock(bool autolock) { m_autolock = autolock; }

protected:

	/** Initialize Object */
	virtual void init();

protected:

	/** Machine to be Evaluated */
	CMachine* m_machine;

	/** Features to be used*/
	CFeatures* m_features;

	/** Labels for the features */
	CLabels* m_labels;

	/** Splitting Strategy to be used */
	CSplittingStrategy* m_splitting_strategy;

	/** Criteria for evaluation */
	CDynamicObjectArray* m_evaluation_criteria;

	/** whether machine will automatically be locked before evaluation */
	bool m_autolock;

	/** whether machine should be unlocked after evaluation */
	bool m_do_unlock;

};

} /* namespace shogun */

#endif /* CMACHINEEVALUATION_H_ */
