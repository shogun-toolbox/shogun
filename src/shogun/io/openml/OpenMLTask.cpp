/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/openml/OpenMLFile.h>
#include <shogun/io/openml/OpenMLTask.h>
#include <shogun/io/openml/utils.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;

struct DatasetStruct
{
	std::string dataset_id;
	std::string target_feature;
};

struct SplitStruct
{
	std::string split_id;
	std::string type;
	std::string data_splits_url;
	std::unordered_map<std::string, std::string> split_parameters;
};

std::shared_ptr<OpenMLTask>
OpenMLTask::get_task(const std::string& task_id, const std::string& api_key)
{
	std::string task_name;
	std::string task_type_id;
	std::unordered_map<std::string, std::string> evaluation_measures;

	DatasetStruct dataset_struct{};
	SplitStruct split_struct{};

	auto reader = OpenMLFile(api_key);
	auto return_string = reader.get("task_file", "json", task_id);

	auto& root = check_response<BACKEND_FORMAT::JSON>(return_string, "task");

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
	auto& json_input = root["input"];

	auto input_array = json_input.GetArray();

	for (const auto& task_settings : input_array)
	{
		if (strcmp(task_settings["name"].GetString(), "source_data") == 0)
		{
			auto dataset_info = task_settings["data_set"].GetObject();
			add_string_to_struct(
			    dataset_info, "data_set_id", dataset_struct.dataset_id);
			add_string_to_struct(
			    dataset_info, "target_feature", dataset_struct.target_feature);
		}
		else if (
		    strcmp(task_settings["name"].GetString(), "estimation_procedure") ==
		    0)
		{
			auto split_info = task_settings["estimation_procedure"].GetObject();
			add_string_to_struct(split_info, "id", split_struct.split_id);
			add_string_to_struct(split_info, "type", split_struct.type);
			add_string_to_struct(
			    split_info, "data_splits_url", split_struct.data_splits_url);

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
			split_struct.split_parameters = split_parameters;

			REQUIRE(
			    split_struct.type == "crossvalidation",
			    "Currently only tasks with cross validation are enabled in "
			    "shogun!\n")
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

	REQUIRE(
	    !dataset_struct.dataset_id.empty(),
	    "The dataset ID is required to retrieve the dataset!\n")
	auto openml_dataset =
	    OpenMLData::get_dataset(dataset_struct.dataset_id, api_key);

	REQUIRE(
	    !split_struct.data_splits_url.empty(),
	    "The split URL is required to retrieve the split information!\n")
	auto openml_split =
	    OpenMLSplit::get_split(split_struct.data_splits_url, api_key);

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

std::vector<std::vector<std::vector<index_t>>>
OpenMLTask::get_train_indices() const
{
	return get_indices(m_split->get_train_idx());
}

std::vector<std::vector<std::vector<index_t>>>
OpenMLTask::get_test_indices() const
{
	return get_indices(m_split->get_test_idx());
}

std::vector<std::vector<std::vector<int32_t>>>
OpenMLTask::get_indices(const std::array<std::vector<int32_t>, 3>& idx) const
{
	// result = (n_repeats, n_folds, ?) where ? is the number of indices in a
	// given fold
	std::vector<std::vector<std::vector<int32_t>>> result(
	    m_split->get_num_repeats(),
	    std::vector<std::vector<int32_t>>(
	        m_split->get_num_folds(), std::vector<int32_t>{}));
	for (int i = 0; i < idx[0].size(); ++i)
	{
		// result[repeat][fold].push_back(data_index)
		result[idx[1][i]][idx[2][i]].push_back(idx[0][i]);
	}
	return result;
}