/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#ifndef __ENSEMBLEMACHINE_H
#define __ENSEMBLEMACHINE_H

#include <algorithm>
#include <memory>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>
#include <shogun/util/traits.h>
#include <shogun/util/zip_iterator.h>
#include <thread>
#include <vector>
namespace shogun
{
	/** EnsembleMachine is a machine that chains multiple machines. It
	 * consists of a sequence of machines as intermediate stages of training
	 * or testing and a CombinationRule as the final stage.
	 */
	class EnsembleMachine : public Machine
	{

	public:
		EnsembleMachine() 
		{
			init();
		};

		template <
		    typename T,
		    typename = std::enable_if_t<traits::is_container<T>::value>>
		EnsembleMachine(const T& machines)
		{
			std::copy(
			    machines.begin(), machines.end(),
			    std::back_inserter(m_machines));
			init();
		}

		~EnsembleMachine() = default;

		void init()
		{
			SG_ADD(&m_machines, "machines", "Array of machines.", ParameterProperties::HYPER);
			SG_ADD(&m_combination_rule, "combination_rule", "Combination rule", ParameterProperties::HYPER);
		}

		void
		set_combination_rule(const std::shared_ptr<CombinationRule>& combination_rule)
		{
			m_combination_rule = combination_rule;
		}

		void add_machine(const std::shared_ptr<Machine>& machine)
		{
			m_machines.push_back(machine);
		}

		bool train_machine(std::shared_ptr<Features> data) override{
			require(m_labels, "Labels not set");
			train(data, m_labels);
			return true;
		}
		void train(
		    const std::shared_ptr<Features>& data,
		    const std::shared_ptr<Labels>& labs)
		{
			const int32_t& num_threads = env()->get_num_threads();
			if (num_threads > 1)
			{
				std::vector<std::thread> threads;
				int32_t num_machine = m_machines.size();
				int32_t machine_per_thread = num_machine / num_threads;
				if (machine_per_thread < 1)
				{
					machine_per_thread = 1;
				}
				for (int t = 0; t < std::min(num_threads, num_machine); t++)
				{
					threads.emplace_back(
					    [&](int32_t start, int32_t end) {
						    for (auto i = start; i < end; i++)
						    {
							    m_machines[i]->set_labels(labs);
							    m_machines[i]->train(data);
						    }
					    },
					    t, t + machine_per_thread);
				}
				for (auto&& thread : threads)
				{
					thread.join();
				}
				for (int i = machine_per_thread * num_threads; i < num_machine; i++)
				{
					m_machines[i]->set_labels(labs);
					m_machines[i]->train(data);
				}
			}
			else
			{
				for (auto&& machine : m_machines)
				{
					machine->set_labels(labs);
					machine->train(data);
				}
			}
		}

		const char* get_name() const override
		{
			return "EnsembleMachine";
		}

		std::shared_ptr<BinaryLabels>
		apply_binary(std::shared_ptr<Features> data) override
		{
			return std::make_shared<BinaryLabels>(apply_vector(data));
		}

		std::shared_ptr<MulticlassLabels>
		apply_multiclass(std::shared_ptr<Features> data) override
		{
			return std::make_shared<MulticlassLabels>(apply_vector(data));
		}

	private:
		SGVector<float64_t> apply_vector(const std::shared_ptr<Features>& data)
		{
			require(m_combination_rule, "Combination Rule not set");
			SGMatrix<float64_t> outputs(
			    data->get_num_vectors(), m_machines.size());
			int col_index = 0;
			for(auto&& machine: m_machines)
			{
				auto vec = machine->apply(data)
							->as<DenseLabels>()->get_labels();
				auto col_begin = outputs.get_column_vector(col_index++);
				std::copy(vec.begin(), vec.end(), col_begin);
			}
			return m_combination_rule->combine(outputs);
		}

		std::shared_ptr<CombinationRule> m_combination_rule;
		std::vector<std::shared_ptr<Machine>> m_machines;
	};

} // namespace shogun
#endif