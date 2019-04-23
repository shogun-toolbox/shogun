/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Fernando Iglesias, Heiko Strathmann, Yuyu Zhang,
 *          Olivier NGuyen, Bjoern Esser, Thoralf Klein, Soeren Sonnenburg,
 *          Soumyajit De
 */

#ifndef BAGGINGMACHINE_H
#define BAGGINGMACHINE_H

#include <shogun/lib/config.h>

#include <shogun/machine/Machine.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/mathematics/RandomNamespace.h>

namespace shogun
{
	class CombinationRule;
	class Evaluation;

	/**
	 * @brief: Bagging algorithm
	 * i.e. bootstrap aggregating
	 */
	class BaggingMachine : public RandomMixin<Machine>
	{
	public:
		/** default ctor */
		BaggingMachine();

		/**
		 * constructor
		 *
		 * @param features training features
		 * @param labels training labels
		 */
		BaggingMachine(std::shared_ptr<Features> features, std::shared_ptr<Labels> labels);

		virtual ~BaggingMachine() = default;

		virtual std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL);
		virtual std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL);
		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL);

		/**
		 * Set number of bags/machine to create
		 *
		 * @param num_bags number of bags
		 */
		void set_num_bags(int32_t num_bags);

		/**
		 * Get number of bags/machines
		 *
		 * @return number of bags
		 */
		int32_t get_num_bags() const;

		/**
		 * Set number of feature vectors to use
		 * for each bag/machine
		 *
		 * @param bag_size number of vectors to use for a bag
		 */
		virtual void set_bag_size(int32_t bag_size);

		/**
		 * Get number of feature vectors that are use
		 * for training each bag/machine
		 *
		 * @return number of vectors used for training for each bag.
		 */
		virtual int32_t get_bag_size() const;

		/**
		 * Get machine for bagging
		 *
		 * @return machine that is being used in bagging
		 */
		std::shared_ptr<Machine> get_machine() const;

		/**
		 * Set machine to use in bagging
		 *
		 * @param machine the machine to use for bagging
		 */
		virtual void set_machine(std::shared_ptr<Machine> machine);

		/**
		 * Set the combination rule to use for aggregating the classification
		 * results
		 *
		 * @param rule combination rule
		 */
		void set_combination_rule(std::shared_ptr<CombinationRule> rule);

		/**
		 * Get the combination rule that is used for aggregating the results
		 *
		 * @return CombinationRule
		 */
		std::shared_ptr<CombinationRule> get_combination_rule() const;

		/** get classifier type
		 *
		 * @return classifier type CT_BAGGING
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_BAGGING;
		}

		/** get out-of-bag error
		 * CombinationRule is used for combining the predictions.
		 *
		 * @param eval Evaluation method to use for calculating the error
		 * @return out-of-bag error.
		 */
		float64_t get_oob_error() const;

		/** name **/
		virtual const char* get_name() const
		{
			return "BaggingMachine";
		}

	protected:
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);

		/**
		 * sets parameters of Machine - useful in Random Forest
		 *
		 * @param m machine
		 * @param idx indices of training vectors chosen in current bag
		 */
		virtual void set_machine_parameters(std::shared_ptr<Machine> m, SGVector<index_t> idx);

		/** helper function for the apply_{regression,..} functions that
		 * computes the output
		 *
		 * @param data the data to compute the output for
		 * @return predictions
		 */
		SGVector<float64_t> apply_get_outputs(std::shared_ptr<Features> data);

		/** helper function for the apply_{binary,..} functions that
		 * computes the output probabilities without combination rules
		 *
		 * @param data the data to compute the output for
		 * @return predictions
		 */
		SGMatrix<float64_t>
			apply_outputs_without_combination(std::shared_ptr<Features> data);

		/** Register paramaters */
		void register_parameters();

		/** Initialize the members with default values */
		void init();

		/**
		 * get the vector of indices for feature vectors that are out of bag
		 *
		 * @param in_bag vector of indices that are in bag.
		 * NOTE: in_bag is a randomly generated with replacement
		 * @return the vector of indices
		 */
		std::shared_ptr<DynamicArray<index_t>>
			get_oob_indices(const SGVector<index_t>& in_bag);

	protected:
		/** bags array */
		std::vector<std::shared_ptr<Machine>> m_bags;

		/** features to train on */
		std::shared_ptr<Features> m_features;

		/** machine to use for bagging */
		std::shared_ptr<Machine> m_machine;

		/** number of bags to create */
		int32_t m_num_bags;

		/** number of vectors to use from the training features */
		int32_t m_bag_size;

		/** combination rule to use */
		std::shared_ptr<CombinationRule> m_combination_rule;

		/** indices of all feature vectors that are out of bag */
		SGVector<bool> m_all_oob_idx;

		/** array of oob indices */
		std::shared_ptr<DynamicObjectArray> m_oob_indices;

		/** metric to calculate the oob error */
		std::shared_ptr<Evaluation> m_oob_evaluation_metric;

#ifndef SWIG
	public:
		static constexpr std::string_view kFeatures = "features";
		static constexpr std::string_view kNBags = "num_bags";
		static constexpr std::string_view kBagSize = "bag_size";
		static constexpr std::string_view kBags = "bags";
		static constexpr std::string_view kCombinationRule = "combination_rule";
		static constexpr std::string_view kAllOobIdx = "all_oob_idx";
		static constexpr std::string_view kOobIndices = "oob_indices";
		static constexpr std::string_view kMachine = "machine";
		static constexpr std::string_view kOobError = "oob_error";
		static constexpr std::string_view kOobEvaluationMetric = "oob_evaluation_metric";
#endif	
	};
} // namespace shogun

#endif /* BAGGINGMACHINE_H */
