/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_OPENMLRUN_H
#define SHOGUN_OPENMLRUN_H

#include <shogun/base/SGObject.h>

#include <shogun/io/openml/OpenMLFlow.h>
#include <shogun/io/openml/OpenMLTask.h>

namespace shogun {
	class OpenMLRun
	{
	public:
		OpenMLRun(
				const std::string& uploader, const std::string& uploader_name,
				const std::string& setup_id, const std::string& setup_string,
				const std::string& parameter_settings,
				std::vector<float64_t> evaluations,
				std::vector<float64_t> fold_evaluations,
				std::vector<float64_t> sample_evaluations,
				const std::string& data_content,
				std::vector<std::string> output_files,
				std::shared_ptr<OpenMLTask> task, std::shared_ptr<OpenMLFlow> flow,
				const std::string& run_id, std::shared_ptr<CSGObject> model,
				std::vector<std::string> tags, std::string predictions_url)
				: m_uploader(uploader), m_uploader_name(uploader_name),
				  m_setup_id(setup_id), m_setup_string(setup_string),
				  m_parameter_settings(parameter_settings),
				  m_evaluations(std::move(evaluations)),
				  m_fold_evaluations(std::move(fold_evaluations)),
				  m_sample_evaluations(std::move(sample_evaluations)),
				  m_data_content(data_content),
				  m_output_files(std::move(output_files)), m_task(std::move(task)),
				  m_flow(std::move(flow)), m_run_id(run_id),
				  m_model(std::move(model)), m_tags(std::move(tags)),
				  m_predictions_url(std::move(predictions_url))
		{
		}

		static std::shared_ptr<OpenMLRun>
		from_filesystem(const std::string& directory);

		static std::shared_ptr<OpenMLRun> run_flow_on_task(
				std::shared_ptr<OpenMLFlow> flow, std::shared_ptr<OpenMLTask> task);

		static std::shared_ptr<OpenMLRun> run_model_on_task(
				std::shared_ptr<CSGObject> model, std::shared_ptr<OpenMLTask> task);

		void to_filesystem(const std::string& directory) const;

		void publish() const;

	private:
		std::string m_uploader;
		std::string m_uploader_name;
		std::string m_setup_id;
		std::string m_setup_string;
		std::string m_parameter_settings;
		std::vector<float64_t> m_evaluations;
		std::vector<float64_t> m_fold_evaluations;
		std::vector<float64_t> m_sample_evaluations;
		std::string m_data_content;
		std::vector<std::string> m_output_files;
		std::shared_ptr<OpenMLTask> m_task;
		std::shared_ptr<OpenMLFlow> m_flow;
		std::string m_run_id;
		std::shared_ptr<CSGObject> m_model;
		std::vector<std::string> m_tags;
		std::string m_predictions_url;
	};
}

#endif //SHOGUN_OPENMLRUN_H
