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
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <vector>
namespace shogun
{

	class EnsembleMachine : public Machine
	{

	public:
		EnsembleMachine() = default;

		template <
		    typename T,
		    typename = std::enable_if_t<traits::is_container<T>::value>>
		EnsembleMachine(const T& machines)
		{
			std::copy(
			    machines.begin(), machines.end(),
			    std::back_inserter(m_machines));
		}

		~EnsembleMachine() = default;
		
		void
		set_combination_rule(std::shared_ptr<CombinationRule> combination_rule)
		{
			m_combination_rule = combination_rule;
		}

		void add_machine(std::shared_ptr<Machine> machine)
		{
			m_machines.push_back(machine);
		}

		void train(
		    const std::shared_ptr<Features>& data,
		    const std::shared_ptr<Labels>& labs)
		{
			for (auto&& machine : m_machines)
			{
				machine->set_labels(labs);
				machine->train(data);
			}
		}

		const char* get_name() const override
		{
			return "EnsembleMachine";
		}

		std::shared_ptr<Labels>
		apply_binary(const std::shared_ptr<Features>& data)
		{
			return std::make_shared<BinaryLabels>(apply_vector(data));
		}

		std::shared_ptr<Labels>
		apply_multiclass(const std::shared_ptr<Features>& data)
		{	
			return std::make_shared<MulticlassLabels>(apply_vector(data));
		}

	private:

        SGVector<float64_t> apply_vector(const std::shared_ptr<Features>& data)
		{
			require(m_combination_rule, "Combination Rule not set");
			SGMatrix<float64_t> outputs(
			    data->get_num_vectors(), m_machines.size());
			auto iter = m_machines.begin();
			for(auto i = 0; i<outputs.num_cols && iter != m_machines.end(); i++)
			{
				auto res = (*iter)->apply(data);
				auto col = outputs.get_column_vector(i);
				auto vec = res-> as<DenseLabels>()->get_labels();
				for(int j = 0; j<outputs.num_rows; j ++)
				{
					col[j] = vec[j];
				}
				iter ++;
			}
			return m_combination_rule->combine(outputs);
		}

		std::shared_ptr<CombinationRule> m_combination_rule;
		std::vector<std::shared_ptr<Machine>> m_machines;
	};

} // namespace shogun
#endif