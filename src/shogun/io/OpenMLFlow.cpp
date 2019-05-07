/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/OpenMLFlow.h>
#include <shogun/lib/type_case.h>
#include <shogun/util/factory.h>

#include <rapidjson/document.h>

#ifdef HAVE_CURL

using namespace shogun;
using namespace rapidjson;

size_t writer(char* data, size_t size, size_t nmemb, std::string* buffer_in)
{
	// adapted from https://stackoverflow.com/a/5780603
	// Is there anything in the buffer?
	if (buffer_in->empty())
	{
		// Append the data to the buffer
		buffer_in->append(data, size * nmemb);

		return size * nmemb;
	}

	return 0;
}

const char* OpenMLReader::xml_server = "https://www.openml.org/api/v1/xml";
const char* OpenMLReader::json_server = "https://www.openml.org/api/v1/json";
const char* OpenMLReader::dataset_description = "/data/{}";
const char* OpenMLReader::list_data_qualities = "/data/qualities/list";
const char* OpenMLReader::data_features = "/data/features/{}";
const char* OpenMLReader::list_dataset_qualities = "/data/qualities/{}";
const char* OpenMLReader::list_dataset_filter = "/data/list/{}";
const char* OpenMLReader::flow_file = "/flow/{}";

const std::unordered_map<std::string, std::string>
    OpenMLReader::m_format_options = {{"xml", xml_server},
                                      {"json", json_server}};
const std::unordered_map<std::string, std::string>
    OpenMLReader::m_request_options = {
        {"dataset_description", dataset_description},
        {"list_data_qualities", list_data_qualities},
        {"data_features", data_features},
        {"list_dataset_qualities", list_dataset_qualities},
        {"list_dataset_filter", list_dataset_filter},
        {"flow_file", flow_file}};

OpenMLReader::OpenMLReader(const std::string& api_key) : m_api_key(api_key)
{
}

void OpenMLReader::openml_curl_request_helper(const std::string& url)
{
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

	openml_curl_error_helper(curl_handle, res);

	curl_easy_cleanup(curl_handle);
}

void OpenMLReader::openml_curl_error_helper(CURL* curl_handle, CURLcode code)
{
	if (code != CURLE_OK)
	{
		// TODO: call curl_easy_cleanup(curl_handle) ?
		SG_SERROR("Curl error: %s\n", curl_easy_strerror(code))
	}
	//	else
	//	{
	//		long response_code;
	//		curl_easy_getinfo(curl_handle, CURLINFO_RESPONSE_CODE,
	//&response_code); 		if (response_code == 200) 			return;
	// else
	//		{
	//			if (response_code == 181)
	//				SG_SERROR("Unknown flow. The flow with the given ID was not
	// found in the database.") 			else if (response_code == 180)
	// SG_SERROR("") 			SG_SERROR("Server code: %d\n", response_code)
	//		}
	//	}
}

#endif // HAVE_CURL

static void check_flow_response(rapidjson::Document& doc)
{
	if (SG_UNLIKELY(doc.HasMember("error")))
	{
		const Value& root = doc["error"];
		SG_SERROR(
		    "Server error %s: %s\n", root["code"].GetString(),
		    root["message"].GetString())
		return;
	}
	REQUIRE(doc.HasMember("flow"), "Unexpected format of OpenML flow.\n");
}

static SG_FORCED_INLINE void emplace_string_to_map(
    const rapidjson::GenericValue<rapidjson::UTF8<char>>& v,
    std::unordered_map<std::string, std::string>& param_dict,
    const std::string& name)
{
	if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
		param_dict.emplace(name, v[name.c_str()].GetString());
	else
		param_dict.emplace(name, "");
}

static SG_FORCED_INLINE void emplace_string_to_map(
    const rapidjson::GenericObject<
        true, rapidjson::GenericValue<rapidjson::UTF8<char>>>& v,
    std::unordered_map<std::string, std::string>& param_dict,
    const std::string& name)
{
	if (v[name.c_str()].GetType() == rapidjson::Type::kStringType)
		param_dict.emplace(name, v[name.c_str()].GetString());
	else
		param_dict.emplace(name, "");
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
	check_flow_response(document);

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
}

void OpenMLFlow::dump()
{
}

std::shared_ptr<OpenMLFlow> OpenMLFlow::from_file()
{
	return std::shared_ptr<OpenMLFlow>();
}

std::shared_ptr<CSGObject> ShogunOpenML::flow_to_model(
    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults)
{
	std::string name;
	std::string val_as_string;
	std::shared_ptr<CSGObject> obj;
	auto params = flow->get_parameters();
	auto components = flow->get_components();
	auto class_name = get_class_info(flow->get_class_name());
	auto module_name = std::get<0>(class_name);
	auto algo_name = std::get<1>(class_name);
	if (module_name == "machine")
		obj = std::shared_ptr<CSGObject>(machine(algo_name));
	else if (module_name == "kernel")
		obj = std::shared_ptr<CSGObject>(kernel(algo_name));
	else if (module_name == "distance")
		obj = std::shared_ptr<CSGObject>(distance(algo_name));
	else
		SG_SERROR("Unsupported factory \"%s\"\n", module_name.c_str())
	auto obj_param = obj->get_params();

	auto put_lambda = [&obj, &name, &val_as_string](const auto& val) {
		// cast value using type from get, i.e. val
		auto val_ = char_to_scalar<std::remove_reference_t<decltype(val)>>(
		    val_as_string.c_str());
		obj->put(name, val_);
	};

	if (initialize_with_defaults)
	{
		for (const auto& param : params)
		{
			Any any_val = obj_param.at(param.first)->get_value();
			name = param.first;
			val_as_string = param.second.at("default_value");
			sg_any_dispatch(any_val, sg_all_typemap, put_lambda);
		}
	}

	for (const auto& component : components)
	{
		CSGObject* a =
		    flow_to_model(component.second, initialize_with_defaults).get();
		//		obj->put(component.first, a);
	}

	return obj;
}

std::shared_ptr<OpenMLFlow>
ShogunOpenML::model_to_flow(const std::shared_ptr<CSGObject>& model)
{
	return std::shared_ptr<OpenMLFlow>();
}

std::tuple<std::string, std::string>
ShogunOpenML::get_class_info(const std::string& class_name)
{
	std::vector<std::string> class_components;
	auto begin = class_name.begin();
	std::tuple<std::string, std::string> result;

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
	if (class_components.size() != 3)
		SG_SERROR("Invalid class name format %s\n", class_name.c_str())
	if (class_components[0] == "shogun")
		result = std::make_tuple(class_components[1], class_components[2]);
	else
		SG_SERROR(
		    "The provided flow is not meant for shogun deserialisation! The "
		    "required library is \"%s\"\n",
		    class_components[0].c_str())

	return result;
}
