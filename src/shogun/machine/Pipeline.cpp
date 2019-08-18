/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/lib/exception/InvalidStateException.h>
#include <shogun/machine/Pipeline.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace shogun
{
	CPipelineBuilder::~CPipelineBuilder()
	{
		for (auto&& stage : m_stages)
		{
			visit([](auto&& object) { SG_UNREF(object) }, stage.second);
		}
	}

	CPipelineBuilder* CPipelineBuilder::over(CTransformer* transformer)
	{
		return over(transformer->get_name(), transformer);
	}

	CPipelineBuilder*
	CPipelineBuilder::over(const std::string& name, CTransformer* transformer)
	{
		require<std::invalid_argument>(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back().second),
		    "Transformers can not be placed after machines. Last element is "
		    "{}",
		    m_stages.back().first);

		SG_REF(transformer);
		m_stages.emplace_back(name, transformer);

		return this;
	}

	CPipeline* CPipelineBuilder::then(CMachine* machine)
	{
		return then(machine->get_name(), machine);
	}

	CPipeline*
	CPipelineBuilder::then(const std::string& name, CMachine* machine)
	{
		require<std::invalid_argument>(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back().second),
		    "Multiple machines are added to pipeline. Last element is {}",
		    m_stages.back().first);

		SG_REF(machine);
		m_stages.emplace_back(name, machine);

		return build();
	}

	CPipelineBuilder*
	CPipelineBuilder::add_stages(std::vector<CSGObject*> stages)
	{
		for (auto stage : stages)
		{
			auto transformer = dynamic_cast<CTransformer*>(stage);
			if (transformer)
			{
				over(transformer);
			}
			else
			{
				auto machine = dynamic_cast<CMachine*>(stage);
				require<std::invalid_argument>(
				    machine, "Stage must be either a "
				             "transformer or a machine. "
				             "Provided {}",
				    stage->get_name());
				SG_REF(machine);
				m_stages.emplace_back(machine->get_name(), machine);
			}
		}
		return this;
	}

	CPipeline* CPipelineBuilder::build()
	{
		check_pipeline();

		auto pipeline = new CPipeline();
		pipeline->m_stages = std::move(m_stages);
		m_stages.clear();

		return pipeline;
	}

	void CPipelineBuilder::check_pipeline() const
	{
		require<InvalidStateException>(
		    !m_stages.empty(), "Pipeline is empty");
		require<InvalidStateException>(
		    holds_alternative<CMachine*>(m_stages.back().second),
		    "Pipline cannot be trained without an "
		    "added machine. Last element "
		    "is {}.",
		    m_stages.back().first);
	}

	CPipeline::CPipeline() : CMachine()
	{
	}

	CPipeline::~CPipeline()
	{
		for (auto&& stage : m_stages)
		{
			visit([](auto&& object) { SG_UNREF(object) }, stage.second);
		}
	}

	bool CPipeline::train_machine(CFeatures* data)
	{
		if (train_require_labels())
		{
			require(m_labels, "No labels given.");
		}
		auto current_data = wrap(data);
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage.second))
			{
				auto transformer = shogun::get<CTransformer*>(stage.second);
				transformer->train_require_labels()
				    ? transformer->fit(current_data, m_labels)
				    : transformer->fit(current_data);

				current_data = wrap(transformer->transform(current_data));
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage.second);
				if (machine->train_require_labels())
					machine->set_labels(m_labels);
				machine->train(current_data);
			}
		}
		return true;
	}

	CLabels* CPipeline::apply(CFeatures* data)
	{
		auto current_data = wrap(data);
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage.second))
			{
				auto transformer = shogun::get<CTransformer*>(stage.second);
				current_data = wrap(transformer->transform(current_data));
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage.second);
				return machine->apply(current_data);
			}
		}

		return nullptr; // unreachable
	}

	bool CPipeline::train_require_labels() const
	{
		bool require_labels = false;

		for (auto&& stage : m_stages)
		{
			visit(
			    [&require_labels](auto&& fittable) {
				    require_labels = fittable->train_require_labels();
				},
			    stage.second);

			if (require_labels)
				return require_labels;
		}

		return require_labels;
	}

	std::string CPipeline::to_string() const
	{
		std::stringstream ss;

		for (auto i : range(m_stages.size()))
		{
			ss << '[' << i << "]: " << m_stages[i].first << std::endl;
		}

		return ss.str();
	}

	CTransformer* CPipeline::get_transformer(const std::string& name) const
	{
		for (auto&& stage : m_stages)
		{
			if (stage.first == name &&
			    holds_alternative<CTransformer*>(stage.second))
				return shogun::get<CTransformer*>(stage.second);
		}

		error<std::invalid_argument>(
		    "Transformer with name {} not found.",
		    name);

		return nullptr;
	}

	CMachine* CPipeline::get_machine() const
	{
		return shogun::get<CMachine*>(m_stages.back().second);
	}

	CSGObject* CPipeline::clone(ParameterProperties pp) const
	{
		auto result = CMachine::clone()->as<CPipeline>();
		for (auto&& stage : m_stages)
		{
			visit(
			    [&](auto object) {
				    result->m_stages.emplace_back(
				        stage.first,
				        object->clone(pp)
				            ->template as<typename std::remove_pointer<decltype(
				                object)>::type>());
				},
			    stage.second);
		}
		return result;
	}

	EProblemType CPipeline::get_machine_problem_type() const
	{
		return get_machine()->get_machine_problem_type();
	}
}
