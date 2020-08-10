/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#ifndef __COMPOSITE_H
#define __COMPOSITE_H
#include <memory>
#include <shogun/ensemble/CombinationRule.h>
#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/machine/EnsembleMachine.h>
#include <shogun/machine/Machine.h>
#include <shogun/transformer/Transformer.h>
#include <vector>
#include <variant>
#include <shogun/base/variant.h>

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace shogun
{
	class Composite : public Machine
	{
	public:
		Composite() = default;

		~Composite() = default;
		std::shared_ptr<Composite> with(const std::shared_ptr<Machine>& machine)
		{
			m_ensemble_machine->add_machine(machine);
			return shared_from_this()->as<Composite>();
		}

		std::shared_ptr<Composite>
		then(const std::shared_ptr<CombinationRule>& rule)
		{
			m_ensemble_machine->set_combination_rule(rule);
			return shared_from_this()->as<Composite>();
		}

		template<typename T>
		void add_stages(T&& stages)
		{
			m_stages = std::forward<T>(stages);
		}
		
		std::shared_ptr<EnsembleMachine> train(
		    const std::shared_ptr<Features>& data,
		    const std::shared_ptr<Labels>& labs)
		{
			std::shared_ptr<Features> current_data = data;
			for(auto& v: m_stages)
			{
				std::visit(overloaded{
					[&](const std::shared_ptr<Transformer>& trans)
					{
						trans->train_require_labels()
				    		? trans->fit(current_data, labs)
				    		: trans->fit(current_data);
						current_data = trans->transform(current_data);
					},
					[](const std::shared_ptr<Machine>& machine)
					{
						error("Machine should not be added in Composite");
					}
				}, v.second);
			}
			m_ensemble_machine->train(current_data, labs);
			return m_ensemble_machine;
		}

	private:
		std::shared_ptr<EnsembleMachine> m_ensemble_machine =
		    std::make_shared<EnsembleMachine>();
		std::vector<std::pair<std::string, 
			variant<std::shared_ptr<Transformer>, std::shared_ptr<Machine>>>>
		    m_stages;
	};
} // namespace shogun
#endif
