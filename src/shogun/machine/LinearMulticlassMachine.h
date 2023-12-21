/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Chiyuan Zhang, Fernando Iglesias,
 *          Evan Shelhamer, Viktor Gal, Soeren Sonnenburg, Yuyu Zhang,
 *          Evangelos Anagnostopoulos
 */

#ifndef _LINEARMULTICLASSMACHINE_H___
#define _LINEARMULTICLASSMACHINE_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/MulticlassMachine.h>

namespace shogun
{

class DotFeatures;
class LinearMachine;
class MulticlassStrategy;

/** @brief generic linear multiclass machine */
class LinearMulticlassMachine : public MulticlassMachine
{
	public:
		/** default constructor  */
		LinearMulticlassMachine() : MulticlassMachine()
		{
			
		}

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param machine linear machine
		 */
		LinearMulticlassMachine(std::shared_ptr<MulticlassStrategy> strategy, std::shared_ptr<Machine> machine ) :
			MulticlassMachine(strategy, machine)
		{
		}

		/** destructor */
		~LinearMulticlassMachine() override
		{
		}

		/** get name */
		const char* get_name() const override
		{
			return "LinearMulticlassMachine";
		}

		virtual int32_t get_num_classes() const {
			return m_num_classes;
		}

		virtual int32_t get_dim_feature_space() const{
			return m_dim_feature_space;
		}

	protected:

		bool train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs) override
		{
			m_num_vectors = data->get_num_vectors();
			m_num_classes = multiclass_labels(labs)->get_num_classes();
			m_dim_feature_space = data->as<DotFeatures>()->get_dim_feature_space();

			require(m_multiclass_strategy, "Multiclass strategy not set");
			int32_t num_classes = labs->as<MulticlassLabels>()->get_num_classes();
   			m_multiclass_strategy->set_num_classes(num_classes);

			m_machines.clear();
			auto train_labels = std::make_shared<BinaryLabels>(get_num_rhs_vectors());
			m_multiclass_strategy->train_start(
				multiclass_labels(labs), train_labels);
			while (m_multiclass_strategy->train_has_more())
			{
				SGVector<index_t> subset=m_multiclass_strategy->train_prepare_next();
				if (subset.vlen)
				{
					train_labels->add_subset(subset);
					data->add_subset(subset);
				}
				m_machine->train(data, train_labels);
				m_machines.push_back(get_machine_from_trained(m_machine));

				if (subset.vlen)
				{
					train_labels->remove_subset();
					data->remove_subset();
				}
			}

			m_multiclass_strategy->train_stop();


			return true;
		}
		/** init machine for train with setting features */
		bool init_machine_for_train(std::shared_ptr<Features> data) override
		{
			require(m_machine, "No machine given in Multiclass constructor");
			return true;
		}

		/** init machines for applying with setting features */
		bool init_machines_for_apply(std::shared_ptr<Features> data) override
		{
			return true;
		}

		/** check features availability */
		bool is_ready() override
		{
			return true;

		}

		/** construct linear machine from given linear machine */
		std::shared_ptr<Machine> get_machine_from_trained(std::shared_ptr<Machine> machine) const override
		{
			return machine->clone(ParameterProperties::MODEL)->as<LinearMachine>();
		}

		/** get number of rhs feature vectors */
		int32_t get_num_rhs_vectors() const override
		{
			return m_num_vectors;
		}

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset instance to set
		 */
		void add_machine_subset(SGVector<index_t> subset) override
		{
			
		}

		/** deletes any subset set to the features of the machine */
		void remove_machine_subset() override
		{
		
		}

	protected:
		int32_t m_num_vectors;
		int32_t m_dim_feature_space;
		int32_t m_num_classes;
};
}
#endif
