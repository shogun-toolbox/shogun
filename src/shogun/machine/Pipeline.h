/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */

#ifndef _PIPELINE_H_
#define _PIPELINE_H_

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

		/** Add a transformer to pipeline.
		 * @param transformer the transformer
		 * @return the current pipeline
		 */
		CPipeline* with(CTransformer* transformer);

		/** Add a machine to pipeline. Pipeline may have only one machine.
		 * @param machine the machine
		 * @return the current pipeline
		 */
		CPipeline* then(CMachine* machine);

		virtual CLabels* apply(CFeatures* data = NULL) override;

		virtual const char* get_name() const override
		{
			return "Pipeline";
		}

		/** List all stages in the pipeline.*/
		void list_stages() const;

		/** Get a transformer in the pipeline.
		 * @param index index of the transformer (starting from zero)
		 * @return the index-th transformer
		 */
		CTransformer* get_transformer(size_t index) const;

		/** Get machine in the pipeline
		 * @return the machine
		 */
		CMachine* get_machine() const;

	protected:
		virtual bool train_machine(CFeatures* data = NULL) override;

		std::vector<variant<CTransformer*, CMachine*>> m_stages;
		virtual bool train_require_labels() const override;
	};
}

#endif
