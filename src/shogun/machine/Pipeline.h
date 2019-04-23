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
	class Pipeline;

	/** @brief Builder of pipeline. */
	class PipelineBuilder : public SGObject
	{
	public:
		virtual ~PipelineBuilder();

		/** Add a transformer with default name to pipeline. The name is
		 * obtained by transformer->get_name().
		 * @param transformer the transformer
		 * @return the current pipeline builder
		 */
		std::shared_ptr<PipelineBuilder> over(std::shared_ptr<Transformer> transformer);

		/** Add a transformer with given name to pipeline
		 * @param name the name of the transformer
		 * @param transformer the transformer
		 * @return the current pipeline builder
		 */
		std::shared_ptr<PipelineBuilder>
		over(const std::string& name, std::shared_ptr<Transformer> transformer);

		/** Add a machine with default name to pipeline. Pipeline may have only
		 * one machine. build() will be called to create a new pipeline
		 * instance.
		 * @param machine the machine
		 * @return the current pipeline
		 */
		std::shared_ptr<Pipeline> then(std::shared_ptr<Machine> machine);

		/** Add a machine with given name to pipeline. Pipeline may have only
		 * one machine. build() will be called to create a new pipeline
		 * instance.
		 * @param name the name of the traansformer
		 * @param machine the machine
		 * @return the current pipeline
		 */
		std::shared_ptr<Pipeline> then(const std::string& name, std::shared_ptr<Machine> machine);

		/** Add a list of stages to pipeline. Stages in the list must be
		 * transformers or machines.
		 * @param stages vector of stages to add
		 * @return the current pipeline
		 */
		std::shared_ptr<PipelineBuilder> add_stages(std::vector<std::shared_ptr<SGObject>> stages);

		/* Build pipeline. A new pipeline instance will be created, current
		 * pipeline builder will become invalid after calling this method.
		 * @return the new pipeline instance
		 */
		std::shared_ptr<Pipeline> build();

		virtual const char* get_name() const override
		{
			return "PipelineBuilder";
		}

	private:
		/** Check pipeline is not empty and machine has been added. */
		void check_pipeline() const;

		std::vector<std::pair<std::string, variant<std::shared_ptr<Transformer>, std::shared_ptr<Machine>>>>
		    m_stages;
	};

	/** Pipeline is a machine that chains multiple transformers and machines. It
	 * consists of a sequence of transformers as intermediate stages of training
	 * or testing and a machine as the final stage. Features are transformed by
	 * transformers and fed into the next stage sequentially.
	 */
	class Pipeline : public Machine
	{
		friend class PipelineBuilder;

	public:
		Pipeline();
		virtual ~Pipeline();

		virtual std::shared_ptr<Labels> apply(std::shared_ptr<Features> data = NULL) override;

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
		std::shared_ptr<Transformer> get_transformer(const std::string& name) const;

		/** Get machine in the pipeline
		 * @return the machine
		 */
		std::shared_ptr<Machine> get_machine() const;

		virtual std::shared_ptr<SGObject> clone(ParameterProperties pp = ParameterProperties::ALL) const override;

		virtual EProblemType get_machine_problem_type() const override;

	protected:
		virtual bool train_machine(std::shared_ptr<Features> data = NULL) override;

		std::vector<std::pair<std::string, variant<std::shared_ptr<Transformer>, std::shared_ptr<Machine>>>>
		    m_stages;
		virtual bool train_require_labels() const override;
	};
}

#endif
