/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Giovanni De Toni, Jacob Walker, Saurabh Mahindre, Yuyu Zhang,
 *          Roman Votyakov
 */

#ifndef __CROSSVALIDATION_H_
#define __CROSSVALIDATION_H_

#include <shogun/lib/config.h>

#include <shogun/evaluation/EvaluationResult.h>
#include <shogun/evaluation/MachineEvaluation.h>
#include <shogun/mathematics/Seedable.h>

namespace shogun
{

	class MachineEvaluation;
	class CrossValidationOutput;
	class CrossValidationStorage;
	class List;

	/** @brief type to encapsulate the results of an evaluation run.
	 */
	class CrossValidationResult : public EvaluationResult
	{
	public:
		CrossValidationResult()
		{
			SG_ADD(&mean, "mean", "Mean of results");
			SG_ADD(
			    &std_dev, "std_dev",
			    "Standard deviation of cross-validation folds");

			mean = 0;
			std_dev = 0;
		}

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 *  @return name of the SGSerializable
		 */
		virtual const char* get_name() const
		{
			return "CrossValidationResult";
		}

		/** print result */
		virtual void print_result()
		{
			io::print("{}+-{}\n", mean, std_dev);
		}

		/**
		 * Get the evaluations mean.
		 * @return mean
		 */
		float64_t get_mean() const
		{
			return mean;
		}

		/**
		 * Get the standard deviation.
		 * @return standard deviation
		 */
		float64_t get_std_dev() const
		{
			return std_dev;
		}

		/**
		 * Set the evaluations mean.
		 * @param mean the mean
		 */
		void set_mean(float64_t ev_mean)
		{
			this->mean = ev_mean;
		}

		/**
		 * Set the standard deviation
		 * @param std_dev the standard deviation
		 */
		void set_std_dev(float64_t ev_std_dev)
		{
			this->std_dev = ev_std_dev;
		}

	private:
		/** mean */
		float64_t mean;
		/** Standard deviation of cross-validation folds */
		float64_t std_dev;
	};

	/** @brief base class for cross-validation evaluation.
	 * Given a learning machine, a splitting strategy, an evaluation criterion,
	 * features and corresponding labels, this provides an interface for
	 * cross-validation. Results may be retrieved using the evaluate method. A
	 * number of repetitions may be specified for obtaining more accurate
	 * results.
	 * The arithmetic mean and standard deviation of different runs is returned.
	 * Default number of runs is one.
	 *
	 * This class calculates an evaluation criterion of every fold and then
	 * calculates the arithmetic mean of all folds. This is for example suitable
	 * for the AUC or for Accuracy. However, for example F1-measure may not be
	 * merged this way (result will be biased). To solve this, different
	 * sub-classes
	 * may average results of each cross validation fold differently by
	 * overwriting
	 * the evaluate_one_run method.
	 *
	 * See [Forman, G. and Scholz, M. (2009). Apples-to-apples in
	 * cross-validation
	 * studies: Pitfalls in classifier performance measurement. Technical
	 * report,
	 * HP Laboratories.] for details on this subject.
	 *
	 */
	class CrossValidation : public Seedable<MachineEvaluation>
	{
	public:
		/** constructor */
		CrossValidation();

		/** constructor
		 * @param machine learning machine to use
		 * @param features features to use for cross-validation
		 * @param labels labels that correspond to the features
		 * @param splitting_strategy splitting strategy to use
		 * @param evaluation_criterion evaluation criterion to use
		 * evaluation
		 */
		CrossValidation(
			std::shared_ptr<Machine> machine, std::shared_ptr<Features> features,
			std::shared_ptr<Labels> labels, std::shared_ptr<SplittingStrategy> splitting_strategy,
			std::shared_ptr<Evaluation> evaluation_criterion);

		/** constructor, for use with custom kernels (no features)
		 * @param machine learning machine to use
		 * @param labels labels that correspond to the features
		 * @param splitting_strategy splitting strategy to use
		 * @param evaluation_criterion evaluation criterion to use
		 */
		CrossValidation(
		    std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels,
		    std::shared_ptr<SplittingStrategy> splitting_strategy,
		    std::shared_ptr<Evaluation> evaluation_criterion);

		/** destructor */
		virtual ~CrossValidation();

		/** setter for the number of runs to use for evaluation */
		void set_num_runs(int32_t num_runs);

		/** @return name of the SGSerializable */
		virtual const char* get_name() const
		{
			return "CrossValidation";
		}

	private:
		void init();

	protected:
		/**
		 * Does the actual evaluation.
		 * @return the cross-validation result
		 */
		virtual std::shared_ptr<EvaluationResult> evaluate_impl() const override;

		/** Evaluates one single cross-validation run.
		 * Current implementation evaluates each fold separately and then
		 * calculates
		 * arithmetic mean. Suitable for accuracy and AUC for example. NOT for
		 * F1-measure. Has to be overridden by sub-classes if results have to be
		 * merged differently
		 *
		 * @return evaluation result of one cross-validation run
		 */
		float64_t evaluate_one_run(int64_t index) const;

		/** number of evaluation runs for one fold */
		int32_t m_num_runs;
	};
}

#endif /* __CROSSVALIDATION_H_ */
