/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SHOGUNOPENML_H
#define SHOGUN_SHOGUNOPENML_H

#include <shogun/base/SGObject.h>
#include <shogun/evaluation/CrossValidationStorage.h>

#include <shogun/io/openml/OpenMLFlow.h>
#include <shogun/io/openml/OpenMLTask.h>

namespace shogun
{
	class OpenMLRun;
	/**
	 * The Shogun OpenML extension to run models from an OpenMLFlow
	 * and convert models to OpenMLFlow.
	 */
	class ShogunOpenML
	{
	public:
		friend class OpenMLRun;
		/**
		 * Instantiates a SGObject from an OpenMLFlow.
		 *
		 * @param flow the flow to instantiate
		 * @param initialize_with_defaults whether to use the default values
		 * specified in the flow
		 * @return the flow as a trainable model
		 */
		static std::shared_ptr<CSGObject> flow_to_model(
		    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults);

		/**
		 * Converts a SGObject to an OpenMLFlow.
		 *
		 * @param model the model to convert
		 * @return the flow from the model conversion
		 */
		static std::shared_ptr<OpenMLFlow>
		model_to_flow(const std::shared_ptr<CSGObject>& model);

	protected:
		static std::unique_ptr<CrossValidationFoldStorage> run_model_on_fold(
				const std::shared_ptr<CMachine>& machine,
				const std::shared_ptr<OpenMLTask>& task,
				const std::shared_ptr<CFeatures>& features,
				const std::shared_ptr<CLabels>& labels,
				const SGVector<index_t>& train_idx,
				const SGVector<index_t>& test_id,
				index_t repeat_number,
				index_t fold_number);

		static std::unique_ptr<CrossValidationFoldStorage> run_model_on_fold(
				const std::shared_ptr<CMachine>& machine,
				const std::shared_ptr<OpenMLTask>& task,
				const std::shared_ptr<CFeatures>& features,
				const std::shared_ptr<CLabels>& labels);

	private:
		/**
		 * Helper function to extract module/factory information from the
		 * class name field of OpenMLFlow. Throws an error either if the
		 * class name field is ill formed (i.e. not
		 * library.module.algorithm) or if the library name is not "shogun".
		 *
		 * @param class_name the flow class_name field
		 * @return a tuple with the module name (factory string) and the
		 * algorithm name
		 */
		static std::pair<std::string, std::string>
		get_class_info(const std::string& class_name);
	};
} // namespace shogun

#endif // SHOGUN_SHOGUNOPENML_H
