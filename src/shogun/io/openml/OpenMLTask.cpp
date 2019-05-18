/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/openml/OpenMLReader.h>
#include <shogun/io/openml/OpenMLTask.h>
#include <shogun/io/openml/utils.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;

std::shared_ptr<OpenMLTask>
OpenMLTask::get_task(const std::string& task_id, const std::string& api_key)
{
	Document document;
	std::string task_name;
	std::string task_type_id;
	std::shared_ptr<OpenMLData> openml_dataset = nullptr;
	std::shared_ptr<OpenMLSplit> openml_split = nullptr;
	std::unordered_map<std::string, std::string> evaluation_measures;

	auto reader = OpenMLReader(api_key);
	auto return_string = reader.get("task_file", "json", task_id);

	document.Parse(return_string.c_str());
	check_response(document, "task");

	const Value& root = document["task"];

	REQUIRE(
	    task_id == root["task_id"].GetString(),
	    "Expected downloaded task to have the same id as the requested task "
	    "id, but got \"%s\", instead of \"%s\".\n",
	    root["task_id"].GetString(), task_id.c_str())

	task_name = root["task_name"].GetString();
	OpenMLTask::TaskType task_type =
	    get_task_from_string(root["task_type"].GetString());
	task_type_id = root["task_type_id"].GetString();

	// expect two elements in input array: dataset and split
	const Value& json_input = root["input"];

	auto input_array = json_input.GetArray();

	for (const auto& task_settings : input_array)
	{
		if (strcmp(task_settings["name"].GetString(), "source_data") == 0)
		{
			auto dataset_info = task_settings["data_set"].GetObject();
			std::string dataset_id = dataset_info["data_set_id"].GetString();
			std::string target_feature =
			    dataset_info["target_feature"].GetString();
			openml_dataset = OpenMLData::get_dataset(dataset_id, api_key);
		}
		else if (
		    strcmp(task_settings["name"].GetString(), "estimation_procedure") ==
		    0)
		{
			auto split_info = task_settings["estimation_procedure"].GetObject();
			std::string split_id = split_info["id"].GetString();
			std::string split_type = split_info["type"].GetString();
			std::string split_url = split_info["data_splits_url"].GetString();
			std::unordered_map<std::string, std::string> split_parameters;
			for (const auto& param : split_info["parameter"].GetArray())
			{
				if (param.HasMember("name") && param.HasMember("value"))
					split_parameters.emplace(
					    param["name"].GetString(), param["value"].GetString());
				else if (param.HasMember("name"))
					split_parameters.emplace(param["name"].GetString(), "");
				else
					SG_SERROR(
					    "Unexpected number of parameters in parameter array "
					    "of estimation_procedure.\n")
			}
			REQUIRE(
			    split_type == "crossvalidation",
			    "Currently only tasks with cross validation are enabled in "
			    "shogun!\n")
			openml_split = OpenMLSplit::get_split(split_url, api_key);
		}
		else if (
		    strcmp(task_settings["name"].GetString(), "evaluation_measures") ==
		    0)
		{
			auto evaluation_info =
			    task_settings["evaluation_measures"].GetObject();
			for (const auto& param : evaluation_info)
			{
				if (param.value.IsString())
					evaluation_measures.emplace(
					    param.name.GetString(), param.value.GetString());
				else
					evaluation_measures.emplace(param.name.GetString(), "");
			}
		}
	}

	if (openml_dataset == nullptr && openml_split == nullptr)
		SG_SERROR("Error parsing task.\n")

	auto result = std::make_shared<OpenMLTask>(
	    task_id, task_name, task_type, task_type_id, evaluation_measures,
	    openml_split, openml_dataset);

	return result;
}

OpenMLTask::TaskType
OpenMLTask::get_task_from_string(const std::string& task_type)
{
	if (task_type == "Supervised Classification")
		return OpenMLTask::TaskType::SUPERVISED_CLASSIFICATION;
	SG_SERROR("OpenMLTask does not support \"%s\"", task_type.c_str())
}

std::vector<std::vector<int64_t>> OpenMLTask::get_train_indices() const
{
	return get_indices(m_split->get_train_idx());
}

std::vector<std::vector<int64_t>> OpenMLTask::get_test_indices() const
{
	return get_indices(m_split->get_test_idx());
}

std::vector<std::vector<int64_t>>
OpenMLTask::get_indices(const std::vector<std::vector<int64_t>>& idx) const
{
	SG_SNOTIMPLEMENTED
	std::vector<std::vector<int64_t>> result;
	return result;
}