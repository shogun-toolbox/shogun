/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#ifndef _PIPELINE_H_
#define _PIPELINE_H_

#include <initializer_list>
#include <shogun/base/variant.h>
#include <shogun/machine/Machine.h>
#include <shogun/transformer/Transformer.h>

namespace shogun
{
	class CPipeline : public CMachine
	{
	public:
		CPipeline();
		virtual ~CPipeline();

		CPipeline* with(CTransformer* transformer);
		CPipeline* then(CMachine* machine);

		virtual CLabels* apply(CFeatures* data = NULL) override;

		virtual const char* get_name() const override
		{
			return "Pipeline";
		}

	protected:
		virtual bool train_machine(CFeatures* data = NULL) override;

		std::vector<variant<CTransformer*, CMachine*>> m_stages;
		virtual bool train_require_labels() const override;
	};
}

#endif
