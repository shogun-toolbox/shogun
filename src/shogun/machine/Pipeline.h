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
#include <utility>

namespace shogun
{
	/** Pipeline is a machine that chains multiple transformers and machines. It
	 * consists of a sequence of transformers as intermediate stages of training
	 * or testing and a machine as the final stage. Features are transformed by
	 * transformers and fed into the next stage sequentially.
	 */
	class CPipeline : public CMachine
	{
	public:
		CPipeline();
		virtual ~CPipeline();

		/** Add a transformer with default name to pipeline. The name is
		 * obtained by transformer->get_name().
		 * @param transformer the transformer
		 * @return the current pipeline
		 */
		CPipeline* with(CTransformer* transformer);

		/** Add a transformer with given name to pipeline
		 * @param name the name of the transformer
		 * @param transformer the transformer
		 * @return the current pipeline
		 */
		CPipeline* with(const std::string& name, CTransformer* transformer);

		/** Add a machine with default name to pipeline. Pipeline may have only
		 * one machine.
		 * @param machine the machine
		 * @return the current pipeline
		 */
		CPipeline* then(CMachine* machine);

		/** Add a machine with given name to pipeline. Pipeline may have only
		 * one machine.
		 * @param name the name of the traansformer
		 * @param machine the machine
		 * @return the current pipeline
		 */
		CPipeline* then(const std::string& name, CMachine* machine);

		virtual CLabels* apply(CFeatures* data = NULL) override;

		virtual const char* get_name() const override
		{
			return "Pipeline";
		}

		/** List all stages in the pipeline.*/
		virtual std::string to_string() const override;

		/** Get a transformer in the pipeline.
		 * @param name name of the transformer
		 * @return the index-th transformer
		 */
		CTransformer* get_transformer(const std::string& name) const;

		/** Get machine in the pipeline
		 * @return the machine
		 */
		CMachine* get_machine() const;

	protected:
		virtual bool train_machine(CFeatures* data = NULL) override;

		std::vector<std::pair<std::string, variant<CTransformer*, CMachine*>>>
		    m_stages;
		virtual bool train_require_labels() const override;

		/** Check pipeline is not empty and machine has been added. */
		void check_pipeline();
	};
}

#endif
