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
	class CPipeline;

	/** @brief Builder of pipeline. */
	class CPipelineBuilder : public CSGObject
	{
	public:
		virtual ~CPipelineBuilder();

		/** Add a transformer with default name to pipeline. The name is
		 * obtained by transformer->get_name().
		 * @param transformer the transformer
		 * @return the current pipeline builder
		 */
		CPipelineBuilder* over(CTransformer* transformer);

		/** Add a transformer with given name to pipeline
		 * @param name the name of the transformer
		 * @param transformer the transformer
		 * @return the current pipeline builder
		 */
		CPipelineBuilder*
		over(const std::string& name, CTransformer* transformer);

		/** Add a machine with default name to pipeline. Pipeline may have only
		 * one machine. build() will be called to create a new pipeline
		 * instance.
		 * @param machine the machine
		 * @return the current pipeline
		 */
		CPipeline* then(CMachine* machine);

		/** Add a machine with given name to pipeline. Pipeline may have only
		 * one machine. build() will be called to create a new pipeline
		 * instance.
		 * @param name the name of the traansformer
		 * @param machine the machine
		 * @return the current pipeline
		 */
		CPipeline* then(const std::string& name, CMachine* machine);

		/** Add a list of stages to pipeline. Stages in the list must be
		 * transformers or machines.
		 * @param stages vector of stages to add
		 * @return the current pipeline
		 */
		CPipelineBuilder* add_stages(std::vector<CSGObject*> stages);

		/* Build pipeline. A new pipeline instance will be created, current
		 * pipeline builder will become invalid after calling this method.
		 * @return the new pipeline instance
		 */
		CPipeline* build();

		virtual const char* get_name() const override
		{
			return "PipelineBuilder";
		}

	private:
		/** Check pipeline is not empty and machine has been added. */
		void check_pipeline() const;

		std::vector<std::pair<std::string, variant<CTransformer*, CMachine*>>>
		    m_stages;
	};

	/** Pipeline is a machine that chains multiple transformers and machines. It
	 * consists of a sequence of transformers as intermediate stages of training
	 * or testing and a machine as the final stage. Features are transformed by
	 * transformers and fed into the next stage sequentially.
	 */
	class CPipeline : public CMachine
	{
		friend class CPipelineBuilder;

	public:
		CPipeline();
		virtual ~CPipeline();

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

		/** Setter for store-model-features-after-training flag. This will be
		 * forwarded to `set_store_model_features` of underlying machines.
		 *
		 * @param store_model whether model should be stored after
		 * training
		 */
		virtual void set_store_model_features(bool store_model) override;

		/** Stores feature data of underlying model. This will be forwarded to
		 * `store_model_features` of underlying machines.
		 */
		virtual void store_model_features() override;

		virtual CSGObject* clone() override;

	protected:
		virtual bool train_machine(CFeatures* data = NULL) override;

		std::vector<std::pair<std::string, variant<CTransformer*, CMachine*>>>
		    m_stages;
		virtual bool train_require_labels() const override;
	};
}

#endif
