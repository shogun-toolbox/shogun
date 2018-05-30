/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#include <shogun/machine/Pipeline.h>

namespace shogun
{
	CPipeline::CPipeline() : CMachine()
	{
	}

	CPipeline::~CPipeline()
	{
		for (auto&& stage : m_stages)
		{
			visit([](auto&& object) { SG_UNREF(object) }, stage);
		}
	}

	CPipeline* CPipeline::with(CTransformer* transformer)
	{
		REQUIRE(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back()),
		    "Transformers can not be placed after machines.\n");

		SG_REF(transformer);
		m_stages.emplace_back(transformer);

		return this;
	}

	CPipeline* CPipeline::then(shogun::CMachine* machine)
	{
		REQUIRE(
		    m_stages.empty() ||
		        holds_alternative<CTransformer*>(m_stages.back()),
		    "Multiple machines are added to pipeline.\n");

		SG_REF(machine);
		m_stages.emplace_back(machine);

		return this;
	}

	bool CPipeline::train_machine(CFeatures* data)
	{
		if (train_require_labels())
		{
			REQUIRE(m_labels, "No labels given.\n");
		}

		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage))
			{
				auto transformer = shogun::get<CTransformer*>(stage);
				transformer->train_require_labels()
				    ? transformer->fit(data, m_labels)
				    : transformer->fit(data);

				data = transformer->transform(data);
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage);
				if (machine->train_require_labels())
					machine->set_labels(m_labels);
				machine->train();
			}
		}
		return true;
	}

	CLabels* CPipeline::apply(CFeatures* data)
	{
		for (auto&& stage : m_stages)
		{
			if (holds_alternative<CTransformer*>(stage))
			{
				auto transformer = shogun::get<CTransformer*>(stage);
				data = transformer->transform(data);
			}
			else
			{
				auto machine = shogun::get<CMachine*>(stage);
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
			    stage);

			if (require_labels)
				return require_labels;
		}

		return require_labels;
	}
}
