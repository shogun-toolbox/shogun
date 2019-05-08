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

/* OpenML server format */
const char* OpenMLReader::xml_server = "https://www.openml.org/api/v1/xml";
const char* OpenMLReader::json_server = "https://www.openml.org/api/v1/json";
/* DATA API */
const char* OpenMLReader::dataset_description = "/data/{}";
const char* OpenMLReader::list_data_qualities = "/data/qualities/list";
const char* OpenMLReader::data_features = "/data/features/{}";
const char* OpenMLReader::list_dataset_qualities = "/data/qualities/{}";
const char* OpenMLReader::list_dataset_filter = "/data/list/{}";
/* FLOW API */
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
		SG_SERROR("Connection error: %s.\n", curl_easy_strerror(code))
	}
}

#endif // HAVE_CURL

/**
 * Checks the returned flow in JSON format
 * @param doc the parsed flow
 */
static void check_flow_response(Document& doc)
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
    const GenericObject<
        true, GenericValue<UTF8<char>>>& v,
    std::unordered_map<std::string, std::string>& param_dict,
    const std::string& name)
{
	if (v[name.c_str()].GetType() == Type::kStringType)
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

/**
 * Class using the Any visitor pattern to convert
 * a string to a C++ type that can be used as a parameter
 * in a Shogun model.
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
		if (!is_null())
		{
			SG_SDEBUG("bool: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
			bool result = strcmp(m_string_val.c_str(), "true") == 0;
			m_model->put(m_parameter, result);
		}
	}
	void on(int32_t* v) final
	{
		if (!is_null())
		{
			SG_SDEBUG("int32: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
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
		if (!is_null())
		{
			SG_SDEBUG("int64: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
			int64_t result = std::stol(m_string_val);
			m_model->put(m_parameter, result);
		}
	}
	void on(float* v) final
	{
		if (!is_null())
		{
			SG_SDEBUG("float: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
			char* end;
			float32_t result = std::strtof(m_string_val.c_str(), &end);
			m_model->put(m_parameter, result);
		}
	}
	void on(double* v) final
	{
		if (!is_null())
		{
			SG_SDEBUG("double: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
			char* end;
			float64_t result = std::strtod(m_string_val.c_str(), &end);
			m_model->put(m_parameter, result);
		}
	}
	void on(long double* v)
	{
		if (!is_null())
		{
			SG_SDEBUG("long double: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
			char* end;
			floatmax_t result = std::strtold(m_string_val.c_str(), &end);
			m_model->put(m_parameter, result);
		}
	}
	void on(CSGObject** v) final
	{
		SG_SDEBUG("CSGObject: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGVector<int>* v) final
	{
		SG_SDEBUG("SGVector<int>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGVector<float>* v) final
	{
		SG_SDEBUG("SGVector<float>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGVector<double>* v) final
	{
		SG_SDEBUG("SGVector<double>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGMatrix<int>* mat) final
	{
		SG_SDEBUG("SGMatrix<int>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGMatrix<float>* mat) final
	{
		SG_SDEBUG("SGMatrix<float>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}
	void on(SGMatrix<double>* mat) final
	{
		SG_SDEBUG("SGMatrix<double>: %s=%s\n", m_parameter.c_str(), m_string_val.c_str())
	}

	bool is_null()
	{
		bool result = strcmp(m_string_val.c_str(), "null") == 0;
		return result;
	}

	void set_parameter_name(const std::string& name)
	{
		m_parameter = name;
	}

	void set_string_value(const std::string& value)
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
	std::shared_ptr<CSGObject> obj;
	if (factory_name == "machine")
		obj = std::shared_ptr<CSGObject>(machine(algo_name));
	else if (factory_name == "kernel")
		obj = std::shared_ptr<CSGObject>(kernel(algo_name));
	else if (factory_name == "distance")
		obj = std::shared_ptr<CSGObject>(distance(algo_name));
	else
		SG_SERROR("Unsupported factory \"%s\".\n", factory_name.c_str())

	return obj;
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
	}
	else if (auto casted_obj = std::dynamic_pointer_cast<CKernel>(nested_obj))
	{
		auto* tmp_clone = dynamic_cast<CKernel*>(casted_obj->clone());
		obj->put(parameter_name, tmp_clone);
	}
	else if (auto casted_obj = std::dynamic_pointer_cast<CDistance>(nested_obj))
	{
		auto* tmp_clone = dynamic_cast<CDistance*>(casted_obj->clone());
		obj->put(parameter_name, tmp_clone);
	}
	else
		SG_SERROR("Could not cast SGObject.\n")
}

std::shared_ptr<CSGObject> ShogunOpenML::flow_to_model(
    std::shared_ptr<OpenMLFlow> flow, bool initialize_with_defaults)
{
	auto params = flow->get_parameters();
	auto components = flow->get_components();
	auto class_name = get_class_info(flow->get_class_name());
	auto module_name = std::get<0>(class_name);
	auto algo_name = std::get<1>(class_name);

	auto obj = instantiate_model_from_factory(module_name, algo_name);
	auto obj_param = obj->get_params();

	std::unique_ptr<StringToShogun> visitor(new StringToShogun(obj));

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
	if (class_components[0] == "shogun")
		result = std::make_tuple(class_components[1], class_components[2]);
	else
		SG_SERROR(
		    "The provided flow is not meant for shogun deserialisation! The "
		    "required library is \"%s\".\n",
		    class_components[0].c_str())
	if (class_components.size() != 3)
		SG_SERROR("Invalid class name format %s.\n", class_name.c_str())

	return result;
}
