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
		LinearMulticlassMachine() : MulticlassMachine(), m_features(NULL)
		{
			SG_ADD(&m_features, "m_features", "Feature object.");
		}

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param features features
		 * @param machine linear machine
		 * @param labs labels
		 */
		LinearMulticlassMachine(std::shared_ptr<MulticlassStrategy> strategy, std::shared_ptr<Features> features, std::shared_ptr<Machine> machine, std::shared_ptr<Labels> labs) :
			MulticlassMachine(strategy, machine,labs), m_features(NULL)
		{
			set_features(features->as<DotFeatures>());
			SG_ADD(&m_features, "m_features", "Feature object.");
		}

		/** destructor */
		virtual ~LinearMulticlassMachine()
		{

		}

		/** get name */
		virtual const char* get_name() const
		{
			return "LinearMulticlassMachine";
		}

		/** set features
		 *
		 * @param f features
		 */
		void set_features(std::shared_ptr<DotFeatures> f)
		{
			m_features = f;
			for (auto m: m_machines)
			{
				auto machine = m->as<LinearMachine>();
				machine->set_features(f);
			}
		}

		/** get features
		 *
		 * @return features
		 */
		std::shared_ptr<DotFeatures> get_features() const
		{
			return m_features;
		}

	protected:

		/** init machine for train with setting features */
		virtual bool init_machine_for_train(std::shared_ptr<Features> data)
		{
			if (!m_machine)
				error("No machine given in Multiclass constructor");

			if (data)
				set_features(data->as<DotFeatures>());

			m_machine->as<LinearMachine>()->set_features(m_features);

			return true;
		}

		/** init machines for applying with setting features */
		virtual bool init_machines_for_apply(std::shared_ptr<Features> data)
		{
			if (data)
				set_features(data->as<DotFeatures>());

			for (auto m: m_machines)
			{
				auto machine = m->as<LinearMachine>();
				ASSERT(m_features)
				ASSERT(machine)
				machine->set_features(m_features);
			}

			return true;
		}

		/** check features availability */
		virtual bool is_ready()
		{
			if (m_features)
				return true;

			return false;
		}

		/** construct linear machine from given linear machine */
		virtual std::shared_ptr<Machine> get_machine_from_trained(std::shared_ptr<Machine> machine) const
		{
			return std::make_shared<LinearMachine>(machine->as<LinearMachine>());
		}

		/** get number of rhs feature vectors */
		virtual int32_t get_num_rhs_vectors() const
		{
			return m_features->get_num_vectors();
		}

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset instance to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset)
		{
			/* changing the subset structure to use subset stacks. This might
			 * have to be revised. Heiko Strathmann */
			m_features->add_subset(subset);
		}

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset()
		{
			/* changing the subset structure to use subset stacks. This might
			 * have to be revised. Heiko Strathmann */
			m_features->remove_subset();
		}

	protected:

		/** features */
		std::shared_ptr<DotFeatures> m_features;
};
}
#endif
