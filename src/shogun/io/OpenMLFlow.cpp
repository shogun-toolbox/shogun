/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/OpenMLFlow.h>
#include <shogun/util/factory.h>
#include <shogun/labels/Labels.h>

#include <rapidjson/document.h>
#ifdef HAVE_CURL
#include "OpenMLFlow.h"
#include <curl/curl.h>

#endif // HAVE_CURL

using namespace shogun;
using namespace rapidjson;

/**
 * The writer callback function used to write the packets to a C++ string.
 * @param data the data received in CURL request
 * @param size always 1
 * @param nmemb the size of data
 * @param buffer_in the buffer to write to
 * @return the size of buffer that was written
 */
size_t writer(char* data, size_t size, size_t nmemb, std::string* buffer_in)
{
	// check that the buffer string points to something
	if (buffer_in != nullptr)
	{
		// Append the data to the buffer
		buffer_in->append(data, size * nmemb);

		return size * nmemb;
	}
	return 0;
}

/* OpenML server format */
const char* OpenMLReader::xml_server = "https://www.openml.org/api/v1/xml";
const char* OpenMLReader::json_server = "https://www.openml.org/api/v1/json";
const char* OpenMLReader::download_server = "";
const char* OpenMLReader::splits_server = "https://www.openml.org/api_splits";

/* DATA API */
const char* OpenMLReader::dataset_description = "/data/{}";
const char* OpenMLReader::list_data_qualities = "/data/qualities/list";
const char* OpenMLReader::data_features = "/data/features/{}";
const char* OpenMLReader::data_qualities = "/data/qualities/{}";
const char* OpenMLReader::list_dataset_qualities = "/data/qualities/{}";
const char* OpenMLReader::list_dataset_filter = "/data/list/{}";
/* FLOW API */
const char* OpenMLReader::flow_file = "/flow/{}";
/* TASK API */
const char* OpenMLReader::task_file = "/task/{}";
/* SPLIT API */
const char* OpenMLReader::get_split = "/get/{}";

const std::unordered_map<std::string, std::string>
    OpenMLReader::m_format_options = {{"xml", xml_server},
                                      {"json", json_server},
                                      {"split", splits_server},
                                      {"download", download_server}};
const std::unordered_map<std::string, std::string>
    OpenMLReader::m_request_options = {
        {"dataset_description", dataset_description},
        {"list_data_qualities", list_data_qualities},
        {"data_features", data_features},
        {"data_qualities", data_qualities},
        {"list_dataset_qualities", list_dataset_qualities},
        {"list_dataset_filter", list_dataset_filter},
        {"flow_file", flow_file},
        {"task_file", task_file}};

void OpenMLReader::openml_curl_request_helper(const std::string& url)
{
#ifdef HAVE_CURL
	CURL* curl_handle = nullptr;

	curl_handle = curl_easy_init();

	if (!curl_handle)
	{
		SG_SERROR("Failed to initialise curl handle.\n")
		return;
	}

	curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl_handle, CURLOPT_HTTPGET, 1);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, writer);
	curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &m_curl_response_buffer);

	CURLcode res = curl_easy_perform(curl_handle);

	if (res != CURLE_OK)
		SG_SERROR("Connection error: %s.\n", curl_easy_strerror(res))

	curl_easy_cleanup(curl_handle);
#endif // HAVE_CURL
}

/**
 * Checks the returned response from OpenML in JSON format
 * @param doc the parsed OpenML JSON format response
 */
static void check_response(const Document& doc, const std::string& type)
{
	if (SG_UNLIKELY(doc.HasMember("error")))
	{
		const Value& root = doc["error"];
		SG_SERROR(
		    "Server error %s: %s\n", root["code"].GetString(),
		    root["message"].GetString())
		return;
	}
	REQUIRE(
	    doc.HasMember(type.c_str()), "Unexpected format of OpenML %s.\n",
	    type.c_str());
}

/**
 * Helper function to add JSON objects as string in map
 * @param v a RapidJSON GenericValue, i.e. string
 * @param param_dict the map to write to
 * @param name the name of the key
 */
static SG_FORCED_INLINE void emplace_string_to_map(
    const GenericValue<UTF8<char>>& v,
    std::unordered_map<std::string, std::string>& param_dict,
    const std::string& name)
{
	if (v[name.c_str()].GetType() == Type::kStringType)
		param_dict.emplace(name, v[name.c_str()].GetString());
	else
		param_dict.emplace(name, "");
}

/**
 * Helper function to add JSON objects as string in map
 * @param v a RapidJSON GenericObject, i.e. array
 * @param param_dict the map to write to
 * @param name the name of the key
 */
static SG_FORCED_INLINE void emplace_string_to_map(
    const GenericObject<true, GenericValue<UTF8<char>>>& v,
    std::unordered_map<std::string, std::string>& param_dict,
    const std::string& name)
{
	if (v[name.c_str()].GetType() == Type::kStringType)
		param_dict.emplace(name, v[name.c_str()].GetString());
	else
		param_dict.emplace(name, "");
}

template <typename T>
SG_FORCED_INLINE T return_if_possible(
    const std::string& name,
    const GenericObject<true, GenericValue<UTF8<char>>>& v)
{
	SG_SNOTIMPLEMENTED
}

template <>
SG_FORCED_INLINE std::string return_if_possible<std::string>(
    const std::string& name,
    const GenericObject<true, GenericValue<UTF8<char>>>& v)
{
	if (v.HasMember(name.c_str()) && v[name.c_str()].IsString())
		return v[name.c_str()].GetString();
	if (v.HasMember(name.c_str()) && !v[name.c_str()].IsString())
		SG_SERROR("Found member \"%s\" but it is not a string", name.c_str())
	if (!v.HasMember(name.c_str()))
		return "";
	SG_SERROR("\"%s\" is not a member of the given object", name.c_str())
	return nullptr;
}

template <>
SG_FORCED_INLINE std::vector<std::string>
return_if_possible<std::vector<std::string>>(
    const std::string& name,
    const GenericObject<true, GenericValue<UTF8<char>>>& v)
{
	std::vector<std::string> result;
	if (!v.HasMember(name.c_str()))
		SG_SERROR("\"%s\" is not a member of the given object", name.c_str())
	if (v[name.c_str()].IsString())
	{
		result.emplace_back(v[name.c_str()].GetString());
	}
	if (v[name.c_str()].IsArray())
	{
		for (const auto& val : v[name.c_str()].GetArray())
		{
			if (val.IsString())
				result.emplace_back(val.GetString());
			else
				SG_SERROR("Found non string member in \"%s\".\n", name.c_str())
		}
	}
	return result;
}

std::shared_ptr<OpenMLFlow> OpenMLFlow::download_flow(
    const std::string& flow_id, const std::string& api_key)
{
	Document document;
	parameters_type params;
	components_type components;
	std::string name;
	std::string description;
	std::string class_name;

	// get flow and parse with RapidJSON
	auto reader = OpenMLReader(api_key);
	auto return_string = reader.get("flow_file", "json", flow_id);
	document.Parse(return_string.c_str());
	check_response(document, "flow");

	// store root for convenience. We know it exists from previous check.
	const Value& root = document["flow"];

	// handle parameters
	if (root.HasMember("parameter"))
	{
		std::unordered_map<std::string, std::string> param_dict;

		if (root["parameter"].IsArray())
		{
			for (const auto& v : root["parameter"].GetArray())
			{
				emplace_string_to_map(v, param_dict, "data_type");
				emplace_string_to_map(v, param_dict, "default_value");
				emplace_string_to_map(v, param_dict, "description");
				params.emplace(v["name"].GetString(), param_dict);
				param_dict.clear();
			}
		}
		else
		{
			// parameter can also be a dict, instead of array
			const auto v = root["parameter"].GetObject();
			emplace_string_to_map(v, param_dict, "data_type");
			emplace_string_to_map(v, param_dict, "default_value");
			emplace_string_to_map(v, param_dict, "description");
			params.emplace(v["name"].GetString(), param_dict);
		}
	}

	// handle components, i.e. kernels
	if (root.HasMember("component"))
	{
		if (root["component"].IsArray())
		{
			for (const auto& v : root["component"].GetArray())
			{
				components.emplace(
				    v["identifier"].GetString(),
				    OpenMLFlow::download_flow(
				        v["flow"]["id"].GetString(), api_key));
			}
		}
		else
		{
			components.emplace(
			    root["component"]["identifier"].GetString(),
			    OpenMLFlow::download_flow(
			        root["component"]["flow"]["id"].GetString(), api_key));
		}
	}

	// get remaining information from flow
	if (root.HasMember("name"))
		name = root["name"].GetString();
	if (root.HasMember("description"))
		description = root["description"].GetString();
	if (root.HasMember("class_name"))
		class_name = root["class_name"].GetString();

	auto flow = std::make_shared<OpenMLFlow>(
	    name, description, class_name, components, params);

	return flow;
}

void OpenMLFlow::upload_flow(const std::shared_ptr<OpenMLFlow>& flow)
{
	SG_SNOTIMPLEMENTED;
}

void OpenMLFlow::dump() const
{
	SG_SNOTIMPLEMENTED;
}

std::shared_ptr<OpenMLFlow> OpenMLFlow::from_file()
{
	SG_SNOTIMPLEMENTED;
	return std::shared_ptr<OpenMLFlow>();
}

std::shared_ptr<OpenMLData>
OpenMLData::get_dataset(const std::string& id, const std::string& api_key)
{
	// description
	Document document;
	auto reader = OpenMLReader(api_key);
	auto return_string = reader.get("dataset_description", "json", id);

	document.Parse(return_string.c_str());
	check_response(document, "data_set_description");

	const Value& dataset_description = document["data_set_description"];

	auto name = return_if_possible<std::string>(
	    "name", dataset_description.GetObject());
	auto description = return_if_possible<std::string>(
	    "description", dataset_description.GetObject());
	auto data_format = return_if_possible<std::string>(
	    "data_format", dataset_description.GetObject());
	auto dataset_id =
	    return_if_possible<std::string>("id", dataset_description.GetObject());
	auto version = return_if_possible<std::string>(
	    "version", dataset_description.GetObject());
	auto creator = return_if_possible<std::string>(
	    "creator", dataset_description.GetObject());
	auto contributor = return_if_possible<std::string>(
	    "contributor", dataset_description.GetObject());
	auto collection_date = return_if_possible<std::string>(
	    "collection_date", dataset_description.GetObject());
	auto upload_date = return_if_possible<std::string>(
	    "upload_date", dataset_description.GetObject());
	auto language = return_if_possible<std::string>(
	    "language", dataset_description.GetObject());
	auto licence = return_if_possible<std::string>(
	    "licence", dataset_description.GetObject());
	auto url =
	    return_if_possible<std::string>("url", dataset_description.GetObject());
	auto default_target_attribute = return_if_possible<std::string>(
	    "default_target_attribute", dataset_description.GetObject());
	auto row_id_attribute = return_if_possible<std::string>(
	    "row_id_attribute", dataset_description.GetObject());
	auto ignore_attribute = return_if_possible<std::string>(
	    "ignore_attribute", dataset_description.GetObject());
	auto version_label = return_if_possible<std::string>(
	    "version_label", dataset_description.GetObject());
	auto citation = return_if_possible<std::string>(
	    "citation", dataset_description.GetObject());
	auto tags = return_if_possible<std::vector<std::string>>(
	    "tag", dataset_description.GetObject());
	auto visibility = return_if_possible<std::string>(
	    "visibility", dataset_description.GetObject());
	auto original_data_url = return_if_possible<std::string>(
	    "original_data_url", dataset_description.GetObject());
	auto paper_url = return_if_possible<std::string>(
	    "paper_url", dataset_description.GetObject());
	auto update_comment = return_if_possible<std::string>(
	    "update_comment", dataset_description.GetObject());
	auto md5_checksum = return_if_possible<std::string>(
	    "md5_checksum", dataset_description.GetObject());

	// features
	std::vector<std::unordered_map<std::string, std::vector<std::string>>>
	    param_vector;
	return_string = reader.get("data_features", "json", id);
	document.Parse(return_string.c_str());
	check_response(document, "data_features");
	const Value& dataset_features = document["data_features"];
	for (const auto& param : dataset_features["feature"].GetArray())
	{
		std::unordered_map<std::string, std::vector<std::string>> param_map;
		for (const auto& param_descriptors : param.GetObject())
		{
			std::vector<std::string> second;
			if (param_descriptors.value.IsArray())
				for (const auto& v : param_descriptors.value.GetArray())
					second.emplace_back(v.GetString());
			else
				second.emplace_back(param_descriptors.value.GetString());

			param_map.emplace(param_descriptors.name.GetString(), second);
		}
		param_vector.push_back(param_map);
	}

	// qualities
	std::vector<std::unordered_map<std::string, std::string>> qualities_vector;
	return_string = reader.get("data_qualities", "json", id);
	document.Parse(return_string.c_str());
	check_response(document, "data_qualities");
	const Value& data_qualities = document["data_qualities"];
	for (const auto& param : data_qualities["quality"].GetArray())
	{
		std::unordered_map<std::string, std::string> param_map;
		for (const auto& param_quality : param.GetObject())
		{
			if (param_quality.name.IsString() && param_quality.value.IsString())
				param_map.emplace(
				    param_quality.name.GetString(),
				    param_quality.value.GetString());
			else if (param_quality.name.IsString())
				param_map.emplace(param_quality.name.GetString(), "");
		}
		qualities_vector.push_back(param_map);
	}

	auto result = std::make_shared<OpenMLData>(
	    name, description, data_format, dataset_id, version, creator,
	    contributor, collection_date, upload_date, language, licence, url,
	    default_target_attribute, row_id_attribute, ignore_attribute,
	    version_label, citation, tags, visibility, original_data_url, paper_url,
	    update_comment, md5_checksum, param_vector, qualities_vector);
	result->set_api_key(api_key);
	return result;
}

std::shared_ptr<CCombinedFeatures> OpenMLData::get_features() noexcept
{
	if (!m_cached_features)
		get_data();
	return m_cached_features;
}

std::shared_ptr<CCombinedFeatures> OpenMLData::get_features(const std::string& label)
{
	auto find_label =
			std::find(m_feature_names.begin(), m_feature_names.end(), label);
	if (find_label == m_feature_names.end())
		SG_SERROR(
			"Requested label \"%s\" not in the dataset!\n", label.c_str())
	if (!m_cached_features)
		get_data();
	auto col_idx = std::distance(m_feature_names.begin(), find_label);
	auto result = std::shared_ptr<CCombinedFeatures>(m_cached_features->clone()->as<CCombinedFeatures>());
	if (result->delete_feature_obj(col_idx))
		SG_SERROR("Error deleting the label column in CombinedFeatures!\n")
	return result;
}

std::shared_ptr<CLabels> OpenMLData::get_labels()
{
	REQUIRE(
	    !m_default_target_attribute.empty(),
	    "A default target attribute is required if no label is given!\n")
	return get_labels(m_default_target_attribute);
}

std::shared_ptr<CLabels> OpenMLData::get_labels(const std::string& label_name)
{
	auto find_label =
	    std::find(m_feature_names.begin(), m_feature_names.end(), label_name);
	if (find_label == m_feature_names.end())
		SG_SERROR(
		    "Requested label \"%s\" not in the dataset!\n", label_name.c_str())
	auto col_idx = std::distance(m_feature_names.begin(), find_label);

	if (!m_cached_features)
		get_data();

	auto target_label_as_feat =
	    std::shared_ptr<CFeatures>(m_cached_features->get_feature_obj(col_idx));

	// TODO: replace with actual enum values
	switch(m_feature_types[col_idx])
	{
		// real features
		case 0:
		{
			auto casted_feat = std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(target_label_as_feat);
			auto labels_vec = casted_feat->get_feature_vector(0);
			auto labels = std::make_shared<CRegressionLabels>();
			labels->set_values(labels_vec);
			return labels;
		} break;
		// nominal features
		case 1:
		{
			auto casted_feat = std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(target_label_as_feat);
			auto labels_vec = casted_feat->get_feature_vector(0);
			auto labels = std::make_shared<CMulticlassLabels>();
			labels->set_values(labels_vec);
			return labels;
		} break;
		default:
			SG_SERROR("Unknown type for label \"%s\"!\n", label_name.c_str())
	}

	return nullptr;
}

void OpenMLData::get_data()
{
	auto reader = OpenMLReader(m_api_key);
	auto return_string = reader.get(m_url);

	// TODO: add ARFF parsing and don't forget feature names and feature types
	m_cached_features = std::make_shared<CCombinedFeatures>();
}

std::shared_ptr<OpenMLSplit>
OpenMLSplit::get_split(const std::string& split_url, const std::string& api_key)
{
	auto reader = OpenMLReader(api_key);
	auto return_string = reader.get("get_split", "split", split_url);

	if (return_string == "Task not providing datasplits.")
		return std::make_shared<OpenMLSplit>();

	auto return_stream = std::istringstream(return_string);
	// TODO: add ARFF parsing here
	// get train/test indices
	// TODO: replace line below with ARFFDeserialiser::get_features()
	auto arff_features = std::make_shared<CCombinedFeatures>();
	REQUIRE(
	    arff_features->get_num_feature_obj() == 4,
	    "Expected a ARFF file with 4 attributes: type, rowid, repeat and "
	    "fold.\n")

	auto train_test_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(0));
	auto rowid_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(1));
	auto repeat_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(2));
	auto fold_feat =
	    std::shared_ptr<CFeatures>(arff_features->get_feature_obj(3));

	auto type_vector = string_feature_to_vector(train_test_feat);
	auto rowid_vector = dense_feature_to_vector(rowid_feat);
	auto repeat_vector = dense_feature_to_vector(repeat_feat);
	auto fold_vector = dense_feature_to_vector(fold_feat);

	std::vector<std::vector<int64_t>> train_idx, test_idx;
	for (int i = 0; i < arff_features->get_num_vectors(); ++i)
	{
		if (type_vector[i] == LabelType::TRAIN)
			train_idx.emplace_back(std::initializer_list<int64_t>{
			    static_cast<int64_t>(rowid_vector[i]),
			    static_cast<int64_t>(repeat_vector[i]),
			    static_cast<int64_t>(fold_vector[i])});
		else
			test_idx.emplace_back(std::initializer_list<int64_t>{
			    static_cast<int64_t>(rowid_vector[i]),
			    static_cast<int64_t>(repeat_vector[i]),
			    static_cast<int64_t>(fold_vector[i])});
	}

	return std::make_shared<OpenMLSplit>(train_idx, test_idx);
}

SGVector<float64_t>
OpenMLSplit::dense_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat =
	    std::dynamic_pointer_cast<CDenseFeatures<float64_t>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CDenseFeatures!\n>");
	return casted_feat->get_feature_vector(0);
}

std::vector<OpenMLSplit::LabelType>
OpenMLSplit::string_feature_to_vector(const std::shared_ptr<CFeatures>& feat)
{
	auto casted_feat = std::dynamic_pointer_cast<CStringFeatures<char>>(feat);
	// this should never happen
	if (!casted_feat)
		SG_SERROR("Error casting a column in the split file from CFeatures to "
		          "CStringFeatures!\n");

	auto to_lower = [](const std::string& line) {
		std::string result;
		std::transform(
		    line.begin(), line.end(), std::back_inserter(result),
		    [](uint8_t val) { return std::tolower(val); });
		return result;
	};

	std::vector<OpenMLSplit::LabelType> result;

	for (int i = 0; i < casted_feat->get_num_vectors(); ++i)
	{
		auto row = casted_feat->get_feature_vector(i);
		std::string label(1, row[0]);
		for (auto j = 1; j < casted_feat->get_max_vector_length(); ++j)
			label.append(1, row[j]);
		if (to_lower(label) == "train")
			result.push_back(LabelType::TRAIN);
		else if (to_lower(label) == "test")
			result.push_back(LabelType::TEST);
		else
			SG_SERROR("Unknown label type in split file %s!\n", label.c_str())
	}
	return result;
}

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
				evaluation_measures.emplace(
				    param.name.GetString(), param.value.GetString());
			}
		}
	}

	if (openml_dataset == nullptr && openml_split == nullptr)
		SG_SERROR("Error parsing task.")

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

SGMatrix<int32_t> OpenMLTask::get_train_indices() const
{
	SG_SNOTIMPLEMENTED
	return SGMatrix<int32_t>();
}

SGMatrix<int32_t> OpenMLTask::get_test_indices() const
{
	SG_SNOTIMPLEMENTED
	return SGMatrix<int32_t>();
}

/**
 * Class using the Any visitor pattern to convert
 * a string to a C++ type that can be used as a parameter
 * in a Shogun model. If the string value is not "null" it will
 * be put in its casted type in the given model with the provided parameter
 * name. If the value is null nothing happens, i.e. no error is thrown
 * and no value is put in model.
 */
class StringToShogun : public AnyVisitor
{
public:
	explicit StringToShogun(std::shared_ptr<CSGObject> model)
	    : m_model(model), m_parameter(""), m_string_val(""){};

	StringToShogun(
	    std::shared_ptr<CSGObject> model, const std::string& parameter,
	    const std::string& string_val)
	    : m_model(model), m_parameter(parameter), m_string_val(string_val){};

	void on(bool* v) final
	{
		SG_SDEBUG("bool: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{
			bool result = strcmp(m_string_val.c_str(), "true") == 0;
			m_model->put(m_parameter, result);
		}
	}
	void on(int32_t* v) final
	{
		SG_SDEBUG("int32: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{
			try
			{
				int32_t result = std::stoi(m_string_val);
				m_model->put(m_parameter, result);
			}
			catch (const std::invalid_argument&)
			{
				// it's an option, i.e. internally represented
				// as an enum but in swig exposed as a string
				m_string_val.erase(
				    std::remove_if(
				        m_string_val.begin(), m_string_val.end(),
				        // remove quotes
				        [](const auto& val) { return val == '\"'; }),
				    m_string_val.end());
				m_model->put(m_parameter, m_string_val);
			}
		}
	}
	void on(int64_t* v) final
	{
		SG_SDEBUG("int64: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{

			int64_t result = std::stol(m_string_val);
			m_model->put(m_parameter, result);
		}
	}
	void on(float* v) final
	{
		SG_SDEBUG("float: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{
			float32_t result = std::stof(m_string_val);
			m_model->put(m_parameter, result);
		}
	}
	void on(double* v) final
	{
		SG_SDEBUG("double: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{
			float64_t result = std::stod(m_string_val);
			m_model->put(m_parameter, result);
		}
	}
	void on(long double* v)
	{
		SG_SDEBUG(
		    "long double: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
		if (!is_null())
		{
			floatmax_t result = std::stold(m_string_val);
			m_model->put(m_parameter, result);
		}
	}
	void on(CSGObject** v) final
	{
		SG_SDEBUG(
		    "CSGObject: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGVector<int>* v) final
	{
		SG_SDEBUG(
		    "SGVector<int>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGVector<float>* v) final
	{
		SG_SDEBUG(
		    "SGVector<float>: %s=%s\n", m_parameter.c_str(),
		    m_string_val.c_str())
	}
	void on(SGVector<double>* v) final
	{
		SG_SDEBUG(
		    "SGVector<double>: %s=%s\n", m_parameter.c_str(),
		    m_string_val.c_str())
	}
	void on(SGMatrix<int>* mat) final
	{
		SG_SDEBUG(
		    "SGMatrix<int>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGMatrix<float>* mat) final
	{
		SG_SDEBUG(
		    "SGMatrix<float>: %s=%s\n", m_parameter.c_str(),
		    m_string_val.c_str())
	}
	void on(SGMatrix<double>* mat) final{SG_SDEBUG(
	    "SGMatrix<double>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())}

	/**
	 * In OpenML "null" is an empty parameter value field.
	 * @return whether the field is "null"
	 */
	SG_FORCED_INLINE bool is_null() const noexcept
	{
		bool result = strcmp(m_string_val.c_str(), "null") == 0;
		return result;
	}

	SG_FORCED_INLINE void set_parameter_name(const std::string& name) noexcept
	{
		m_parameter = name;
	}

	SG_FORCED_INLINE void set_string_value(const std::string& value) noexcept
	{
		m_string_val = value;
	}

private:
	std::shared_ptr<CSGObject> m_model;
	std::string m_parameter;
	std::string m_string_val;
};

/**
 * Instantiates a CSGObject using a factory
 * @param factory_name the name of the factory
 * @param algo_name the name of algorithm passed to factory
 * @return the instantiated object using a factory
 */
std::shared_ptr<CSGObject> instantiate_model_from_factory(
    const std::string& factory_name, const std::string& algo_name)
{
	if (factory_name == "machine")
		return std::shared_ptr<CSGObject>(machine(algo_name));
	if (factory_name == "kernel")
		return std::shared_ptr<CSGObject>(kernel(algo_name));
	if (factory_name == "distance")
		return std::shared_ptr<CSGObject>(distance(algo_name));

	SG_SERROR("Unsupported factory \"%s\".\n", factory_name.c_str())

	return nullptr;
}

/**
 * Downcasts a CSGObject and puts it in the map of obj.
 * @param obj the main object
 * @param nested_obj the object to be casted and put in the obj map.
 * @param parameter_name the name of nested_obj
 */
void cast_and_put(
    const std::shared_ptr<CSGObject>& obj,
    const std::shared_ptr<CSGObject>& nested_obj,
    const std::string& parameter_name)
{
	if (auto casted_obj = std::dynamic_pointer_cast<CMachine>(nested_obj))
	{
		// TODO: remove clone
		//  temporary fix until shared_ptr PR merged
		auto* tmp_clone = dynamic_cast<CMachine*>(casted_obj->clone());
		obj->put(parameter_name, tmp_clone);
		return;
	}
	if (auto casted_obj = std::dynamic_pointer_cast<CKernel>(nested_obj))
	{
		auto* tmp_clone = dynamic_cast<CKernel*>(casted_obj->clone());
		obj->put(parameter_name, tmp_clone);
		return;
	}
	if (auto casted_obj = std::dynamic_pointer_cast<CDistance>(nested_obj))
	{
		auto* tmp_clone = dynamic_cast<CDistance*>(casted_obj->clone());
		obj->put(parameter_name, tmp_clone);
		return;
	}
	SG_SERROR("Could not cast SGObject.\n")
}

std::shared_ptr<CSGObject> ShogunOpenML::flow_to_model(
    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults)
{
	auto params = flow->get_parameters();
	auto components = flow->get_components();
	auto class_name = get_class_info(flow->get_class_name());
	auto module_name = class_name.first;
	auto algo_name = class_name.second;

	auto obj = instantiate_model_from_factory(module_name, algo_name);
	auto obj_param = obj->get_params();

	auto visitor = std::make_unique<StringToShogun>(obj);

	if (initialize_with_defaults)
	{
		for (const auto& param : params)
		{
			Any any_val = obj_param.at(param.first)->get_value();
			std::string name = param.first;
			std::string val_as_string = param.second.at("default_value");
			visitor->set_parameter_name(name);
			visitor->set_string_value(val_as_string);
			any_val.visit(visitor.get());
		}
	}

	for (const auto& component : components)
	{
		std::shared_ptr<CSGObject> nested_obj =
		    flow_to_model(component.second, initialize_with_defaults);
		cast_and_put(obj, nested_obj, component.first);
	}

	SG_SDEBUG("Final object: %s.\n", obj->to_string().c_str());

	return obj;
}

std::shared_ptr<OpenMLFlow>
ShogunOpenML::model_to_flow(const std::shared_ptr<CSGObject>& model)
{
	return std::shared_ptr<OpenMLFlow>();
}

std::pair<std::string, std::string>
ShogunOpenML::get_class_info(const std::string& class_name)
{
	std::vector<std::string> class_components;
	auto begin = class_name.begin();
	std::pair<std::string, std::string> result;

	for (auto it = class_name.begin(); it != class_name.end(); ++it)
	{
		if (*it == '.')
		{
			class_components.emplace_back(std::string(begin, it));
			begin = std::next(it);
		}
		if (std::next(it) == class_name.end())
			class_components.emplace_back(std::string(begin, std::next(it)));
	}

	if (class_components[0] == "shogun" && class_components.size() == 3)
		result = std::make_pair(class_components[1], class_components[2]);
	else if (class_components[0] == "shogun" && class_components.size() != 3)
		SG_SERROR("Invalid class name format %s.\n", class_name.c_str())
	else
		SG_SERROR(
		    "The provided flow is not meant for shogun deserialisation! The "
		    "required library is \"%s\".\n",
		    class_components[0].c_str())

	return result;
}

CLabels* ShogunOpenML::run_model_on_fold(
    const std::shared_ptr<CSGObject>& model,
    const std::shared_ptr<OpenMLTask>& task, CFeatures* X_train,
    index_t repeat_number, index_t fold_number, CLabels* y_train,
    CFeatures* X_test)
{
	auto task_type = task->get_task_type();
	auto model_clone = std::shared_ptr<CSGObject>(model->clone());

	switch (task_type)
	{
	case OpenMLTask::TaskType::SUPERVISED_CLASSIFICATION:
	case OpenMLTask::TaskType::SUPERVISED_REGRESSION:
	{
		if (auto machine = std::dynamic_pointer_cast<CMachine>(model_clone))
		{
			machine->put("labels", y_train);
			machine->train(X_train);
			return machine->apply(X_test);
		}
		else
			SG_SERROR("The provided model is not a trainable machine!\n")
	}
	break;
	case OpenMLTask::TaskType::LEARNING_CURVE:
		SG_SNOTIMPLEMENTED
	case OpenMLTask::TaskType::SUPERVISED_DATASTREAM_CLASSIFICATION:
		SG_SNOTIMPLEMENTED
	case OpenMLTask::TaskType::CLUSTERING:
		SG_SNOTIMPLEMENTED
	case OpenMLTask::TaskType::MACHINE_LEARNING_CHALLENGE:
		SG_SNOTIMPLEMENTED
	case OpenMLTask::TaskType::SURVIVAL_ANALYSIS:
		SG_SNOTIMPLEMENTED
	case OpenMLTask::TaskType::SUBGROUP_DISCOVERY:
		SG_SNOTIMPLEMENTED
	}
	return nullptr;
}

std::shared_ptr<OpenMLRun> OpenMLRun::run_model_on_task(
    std::shared_ptr<CSGObject> model, std::shared_ptr<OpenMLTask> task)
{
	SG_SNOTIMPLEMENTED
	return std::shared_ptr<OpenMLRun>();
}

std::shared_ptr<OpenMLRun> OpenMLRun::run_flow_on_task(
    std::shared_ptr<OpenMLFlow> flow, std::shared_ptr<OpenMLTask> task)
{
	auto data = task->get_dataset();
	std::shared_ptr<CFeatures> train_features, test_features;
	std::shared_ptr<CLabels> train_labels, test_labels;

	if (task->get_split()->contains_splits())
		SG_SNOTIMPLEMENTED
	else
	{
		auto labels = data->get_labels();
		auto feat = data->get_features();
	}
	return std::shared_ptr<OpenMLRun>();
}

std::shared_ptr<OpenMLRun>
OpenMLRun::from_filesystem(const std::string& directory)
{
	SG_SNOTIMPLEMENTED
	return nullptr;
}

void OpenMLRun::to_filesystem(const std::string& directory) const
{
	SG_SNOTIMPLEMENTED
}

void OpenMLRun::publish() const
{
	SG_SNOTIMPLEMENTED
}
