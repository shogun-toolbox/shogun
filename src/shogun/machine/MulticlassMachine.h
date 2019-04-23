/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Chiyuan Zhang, Soeren Sonnenburg,
 *          Fernando Iglesias, Jiaolong Xu, Heiko Strathmann, Evan Shelhamer,
 *          Yuyu Zhang, Shell Hu, Thoralf Klein, Bjoern Esser
 */

#ifndef _MULTICLASSMACHINE_H___
#define _MULTICLASSMACHINE_H___

#include <shogun/lib/config.h>

#include <shogun/machine/BaseMulticlassMachine.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/multiclass/MulticlassStrategy.h>

#include <shogun/util/converters.h>

namespace shogun
{

class Features;
class Labels;
class MulticlassLabels;
class MultilabelLabels;

/** @brief experimental abstract generic multiclass machine class */
class MulticlassMachine : public BaseMulticlassMachine
{
	public:
		/** default constructor  */
		MulticlassMachine();

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param machine machine
		 * @param labels labels
		 */
		MulticlassMachine(std::shared_ptr<MulticlassStrategy> strategy, std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labels);

		/** destructor */
		virtual ~MulticlassMachine();

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(std::shared_ptr<Labels> lab);

		/** set machine
		 *
		 * @param num index of machine
		 * @param machine machine to set
		 * @return if setting was successful
		 */
		inline bool set_machine(int32_t num, std::shared_ptr<Machine> machine)
		{
			ASSERT(num<utils::safe_convert<int32_t>(m_machines.size()) && num>=0)
			if (machine != NULL && !is_acceptable_machine(machine))
				error("Machine {} is not acceptable by {}", machine->get_name(), this->get_name());

			m_machines.insert(m_machines.begin()+num, machine);
			return true;
		}

		/** get machine
		 *
		 * @param num index of machine to get
		 * @return SVM at number num
		 */
		inline std::shared_ptr<Machine> get_machine(int32_t num) const
		{
			return m_machines.at(num);
		}

		/** get outputs of i-th submachine
		 * @param i number of submachine
		 * @return outputs
		 */
		virtual std::shared_ptr<BinaryLabels> get_submachine_outputs(int32_t i);

		/** get output of i-th submachine for num-th vector
		 * @param i number of submachine
		 * @param num number of feature vector
		 * @return output
		 */
		virtual float64_t get_submachine_output(int32_t i, int32_t num);

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL);

		/** classify all examples with multiple output
		 *
		 * @return resulting labels
		 */
		virtual std::shared_ptr<MultilabelLabels> apply_multilabel_output(std::shared_ptr<Features> data=NULL, int32_t n_outputs=5);

		/** classify one example
		 * @param vec_idx
		 * @return label
		 */
		virtual float64_t apply_one(int32_t vec_idx);

		/** get the type of multiclass'ness
		 *
		 * @return multiclass type one vs one etc
		 */
		inline std::shared_ptr<MulticlassStrategy> get_multiclass_strategy() const
		{

			return m_multiclass_strategy;
		}

		/** returns rejection strategy
		 *
		 * @return rejection strategy
		 */
		inline std::shared_ptr<RejectionStrategy> get_rejection_strategy() const
		{
			return m_multiclass_strategy->get_rejection_strategy();
		}

		/** sets rejection strategy
		 *
		 * @param rejection_strategy rejection strategy to be set
		 */
		inline void set_rejection_strategy(std::shared_ptr<RejectionStrategy> rejection_strategy)
		{
			m_multiclass_strategy->set_rejection_strategy(rejection_strategy);
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassMachine";
		}

		/** get prob output heuristic of multiclass strategy */
		inline EProbHeuristicType get_prob_heuris()
		{
			return m_multiclass_strategy->get_prob_heuris_type();
		}

		/** set prob output heuristic of multiclass strategy
		 * @param prob_heuris type of probability heuristic
		 */
		inline void set_prob_heuris(EProbHeuristicType prob_heuris)
		{
			m_multiclass_strategy->set_prob_heuris_type(prob_heuris);
		}

	protected:
		/** init strategy */
		void init_strategy();

		/** clear machines */
		void clear_machines();

		/** train machine */
		virtual bool train_machine(std::shared_ptr<Features> data = NULL);

		/** abstract init machine for training method */
		virtual bool init_machine_for_train(std::shared_ptr<Features> data) = 0;

		/** abstract init machines for applying method */
		virtual bool init_machines_for_apply(std::shared_ptr<Features> data) = 0;

		/** check whether machine is ready */
		virtual bool is_ready() = 0;

		/** obtain machine from trained one */
		virtual std::shared_ptr<Machine> get_machine_from_trained(std::shared_ptr<Machine> machine) const = 0;

		/** get num rhs vectors */
		virtual int32_t get_num_rhs_vectors() const = 0;

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset) = 0;

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset() = 0;

		/** whether the machine is acceptable in set_machine */
		virtual bool is_acceptable_machine(std::shared_ptr<Machine >machine)
		{
			return true;
		}

	private:

		/** register parameters */
		void register_parameters();

	protected:
		/** type of multiclass strategy */
		std::shared_ptr<MulticlassStrategy >m_multiclass_strategy;

		/** machine */
		std::shared_ptr<Machine> m_machine;
};
}
#endif
