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
	PipelineBuilder::~PipelineBuilder()
	{
	}

	std::shared_ptr<PipelineBuilder> PipelineBuilder::over(std::shared_ptr<Transformer> transformer)
	{
		return over(transformer->get_name(), transformer);
	}

	std::shared_ptr<PipelineBuilder>
	PipelineBuilder::over(const std::string& name, std::shared_ptr<Transformer> transformer)
	{
		REQUIRE_E(
		    m_stages.empty() ||
		        holds_alternative<std::shared_ptr<Transformer>>(m_stages.back().second),
		    std::invalid_argument,
		    "Transformers can not be placed after machines. Last element is "
		    "%s\n",
		    m_stages.back().first.c_str());


		m_stages.emplace_back(name, transformer);

		return shared_from_this()->as<PipelineBuilder>();
	}

	std::shared_ptr<Pipeline> PipelineBuilder::then(std::shared_ptr<Machine> machine)
	{
		return then(machine->get_name(), machine);
	}

	std::shared_ptr<Pipeline>
	PipelineBuilder::then(const std::string& name, std::shared_ptr<Machine> machine)
	{
		REQUIRE_E(
		    m_stages.empty() ||
		        holds_alternative<std::shared_ptr<Transformer>>(m_stages.back().second),
		    std::invalid_argument,
		    "Multiple machines are added to pipeline. Last element is %s\n",
		    m_stages.back().first.c_str());


		m_stages.emplace_back(name, machine);

		return build();
	}

	std::shared_ptr<PipelineBuilder>
	PipelineBuilder::add_stages(std::vector<std::shared_ptr<SGObject>> stages)
	{
		for (auto stage : stages)
		{
			auto transformer = stage->as<Transformer>();
			if (transformer)
			{
				over(transformer);
			}
			else
			{
				auto machine = stage->as<Machine>();
				REQUIRE_E(
				    machine, std::invalid_argument, "Stage must be either a "
				                                    "transformer or a machine. "
				                                    "Provided %s\n",
				    stage->get_name());

				m_stages.emplace_back(machine->get_name(), machine);
			}
		}
		return shared_from_this()->as<PipelineBuilder>();
	}

	std::shared_ptr<Pipeline> PipelineBuilder::build()
	{
		check_pipeline();

		auto pipeline = std::make_shared<Pipeline>();
		pipeline->m_stages = std::move(m_stages);
		m_stages.clear();

		return pipeline;
	}

	void PipelineBuilder::check_pipeline() const
	{
		REQUIRE_E(
		    !m_stages.empty(), InvalidStateException, "Pipeline is empty");
		REQUIRE_E(
		    holds_alternative<std::shared_ptr<Machine>>(m_stages.back().second),
		    InvalidStateException, "Pipline cannot be trained without an "
		                           "added machine. Last element "
		                           "is %s.\n",
		    m_stages.back().first.c_str());
	}

	Pipeline::Pipeline() : Machine()
	{
	}

	Pipeline::~Pipeline()
	{
	}

	bool Pipeline::train_machine(std::shared_ptr<Features> data)
	{
		if (train_require_labels())
		{
			REQUIRE(m_labels, "No labels given.\n");
		}
		auto current_data = data;
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<std::shared_ptr<Transformer>>(stage.second))
			{
				auto transformer = shogun::get<std::shared_ptr<Transformer>>(stage.second);
				transformer->train_require_labels()
				    ? transformer->fit(current_data, m_labels)
				    : transformer->fit(current_data);

				current_data = transformer->transform(current_data);
			}
			else
			{
				auto machine = shogun::get<std::shared_ptr<Machine>>(stage.second);
				if (machine->train_require_labels())
					machine->set_labels(m_labels);
				machine->train(current_data);
			}
		}
		return true;
	}

	std::shared_ptr<Labels> Pipeline::apply(std::shared_ptr<Features> data)
	{
		auto current_data = data;
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<std::shared_ptr<Transformer>>(stage.second))
			{
				auto transformer = shogun::get<std::shared_ptr<Transformer>>(stage.second);
				current_data = transformer->transform(current_data);
			}
			else
			{
				auto machine = shogun::get<std::shared_ptr<Machine>>(stage.second);
				return machine->apply(current_data);
			}
		}

		return nullptr; // unreachable
	}

	bool Pipeline::train_require_labels() const
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

	std::string Pipeline::to_string() const
	{
		std::stringstream ss;

		for (auto i : range(m_stages.size()))
		{
			ss << '[' << i << "]: " << m_stages[i].first << std::endl;
		}

		return ss.str();
	}

	std::shared_ptr<Transformer> Pipeline::get_transformer(const std::string& name) const
	{
		for (auto&& stage : m_stages)
		{
			if (stage.first == name &&
			    holds_alternative<std::shared_ptr<Transformer>>(stage.second))
				return shogun::get<std::shared_ptr<Transformer>>(stage.second);
		}

		SG_THROW(
		    std::invalid_argument, "Transformer with name %s not found.\n",
		    name.c_str());

		return nullptr;
	}

	std::shared_ptr<Machine> Pipeline::get_machine() const
	{
		return shogun::get<std::shared_ptr<Machine>>(m_stages.back().second);
	}

	void Pipeline::set_store_model_features(bool store_model)
	{
		get_machine()->set_store_model_features(store_model);
	}

	void Pipeline::store_model_features()
	{
		get_machine()->store_model_features();
	}

	std::shared_ptr<SGObject> Pipeline::clone() const
	{
		auto result = Machine::clone()->as<Pipeline>();
		for (auto&& stage : m_stages)
		{
			visit(
			    [&](auto object) {
				    result->m_stages.emplace_back(
				        stage.first,
				        object->clone()->template as<
				        	typename std::remove_pointer_t<decltype(object.get())>>()
				        );
				},
			    stage.second);
		}
		return result;
	}

	EProblemType Pipeline::get_machine_problem_type() const
	{
		return get_machine()->get_machine_problem_type();
	}
}
