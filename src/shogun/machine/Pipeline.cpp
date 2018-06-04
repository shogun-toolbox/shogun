/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/machine/Pipeline.h>
#include <sstream>
#include <string>

namespace shogun
{
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

	CPipeline* CPipeline::with(CTransformer* transformer)
	{
		return with(transformer->get_name(), transformer);
	}

	CPipeline*
	CPipeline::with(const std::string& name, CTransformer* transformer)
	{
		REQUIRE(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back().second),
		    "Transformers can not be placed after machines.\n");

		SG_REF(transformer);
		m_stages.emplace_back(name, transformer);

		return this;
	}

	CPipeline* CPipeline::then(CMachine* machine)
	{
		return then(machine->get_name(), machine);
	}

	CPipeline* CPipeline::then(const std::string& name, CMachine* machine)
	{
		REQUIRE(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back().second),
		    "Multiple machines are added to pipeline.\n");

		SG_REF(machine);
		m_stages.emplace_back(name, machine);

		return this;
	}

	bool CPipeline::train_machine(CFeatures* data)
	{
		REQUIRE(
		    !m_stages.empty() &&
		        holds_alternative<CMachine*>(m_stages.back().second),
		    "Machine has not been added.\n");
		if (train_require_labels())
		{
			REQUIRE(m_labels, "No labels given.\n");
		}

		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage.second))
			{
				auto transformer = shogun::get<CTransformer*>(stage.second);
				transformer->train_require_labels()
				    ? transformer->fit(data, m_labels)
				    : transformer->fit(data);

				data = transformer->transform(data);
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage.second);
				if (machine->train_require_labels())
					machine->set_labels(m_labels);
				machine->train(data);
			}
		}
		return true;
	}

	CLabels* CPipeline::apply(CFeatures* data)
	{
		REQUIRE(
		    !m_stages.empty() &&
		        holds_alternative<CMachine*>(m_stages.back().second),
		    "Machine has not been added.\n");
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage.second))
			{
				auto transformer = shogun::get<CTransformer*>(stage.second);
				data = transformer->transform(data);
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage.second);
				return machine->apply(data);
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

		SG_ERROR("Transformer with name %s not found.\n", name.c_str());

		return nullptr;
	}

	CMachine* CPipeline::get_machine() const
	{
		REQUIRE(
		    !m_stages.empty() &&
		        holds_alternative<CMachine*>(m_stages.back().second),
		    "Machine has not been added.\n");
		return shogun::get<CMachine*>(m_stages.back().second);
	}
}
