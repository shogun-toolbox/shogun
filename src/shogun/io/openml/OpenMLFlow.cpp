/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/openml/OpenMLFlow.h>

#include <shogun/io/openml/OpenMLReader.h>
#include <shogun/io/openml/utils.h>

#include <rapidjson/document.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;


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