/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Heiko Strathmann, Sergey Lisitsyn, Yuyu Zhang,
 *          Soeren Sonnenburg, Giovanni De Toni
 */

#ifndef CMACHINEEVALUATION_H_
#define CMACHINEEVALUATION_H_

#include <shogun/evaluation/Evaluation.h>
#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/lib/StoppableSGObject.h>
#include <shogun/lib/config.h>

namespace shogun
{
	class Machine;
	class Features;
	class Labels;
	class SplittingStrategy;
	class Evaluation;

	/** @brief Machine Evaluation is an abstract class
	 * that evaluates a machine according to some criterion.
	 *
	 */
	class MachineEvaluation : public StoppableSGObject
	{

	public:
		MachineEvaluation();

		/** constructor
		 * @param machine learning machine to use
		 * @param features features to use for cross-validation
		 * @param labels labels that correspond to the features
		 * @param splitting_strategy splitting strategy to use
		 * @param evaluation_criterion evaluation criterion to use
		 * @param autolock whether machine should be auto-locked before
		 * evaluation
		 */
		MachineEvaluation(
		    std::shared_ptr<Machine> machine, std::shared_ptr<Features> features, std::shared_ptr<Labels> labels,
		    std::shared_ptr<SplittingStrategy> splitting_strategy,
		    std::shared_ptr<Evaluation> evaluation_criterion, bool autolock = true);

		/** constructor, for use with custom kernels (no features)
		 * @param machine learning machine to use
		 * @param labels labels that correspond to the features
		 * @param splitting_strategy splitting strategy to use
		 * @param evaluation_criterion evaluation criterion to use
		 * @param autolock autolock
		 */
		MachineEvaluation(
		    std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels,
		    std::shared_ptr<SplittingStrategy> splitting_strategy,
		    std::shared_ptr<Evaluation> evaluation_criterion, bool autolock = true);

		virtual ~MachineEvaluation();

		/** @return in which direction is the best evaluation value? */
		EEvaluationDirection get_evaluation_direction() const;

		/** method for evaluation. Performs cross-validation.
		 * Is repeated m_num_runs. If this number is larger than one, a
		 * confidence
		 * interval is calculated if m_conf_int_alpha is (0<p<1).
		 * By default m_num_runs=1 and m_conf_int_alpha=0
		 *
		 * @return result of evaluation (already SG_REF'ed)
		 */
		virtual std::shared_ptr<EvaluationResult> evaluate() const;

		/** @return underlying learning machine */
		std::shared_ptr<Machine> get_machine() const;

	protected:
		/** Initialize Object */
		virtual void init();

		/**
		 * Implementation of the evaluation procedure. Called
		 * by evaluate() method. This method has to SG_REF its result
		 * before returning it.
		 * @return the evaluation result
		 */
		virtual std::shared_ptr<EvaluationResult> evaluate_impl() const = 0;

		/** connect the machine instance to the signal handler */
	protected:
		/** Machine to be Evaluated */
		std::shared_ptr<Machine> m_machine;

		/** Features to be used*/
		std::shared_ptr<Features> m_features;

		/** Labels for the features */
		std::shared_ptr<Labels> m_labels;

		/** Splitting Strategy to be used */
		std::shared_ptr<SplittingStrategy> m_splitting_strategy;

		/** Criterion for evaluation */
		std::shared_ptr<Evaluation> m_evaluation_criterion;
	};

} /* namespace shogun */

#endif /* CMACHINEEVALUATION_H_ */
