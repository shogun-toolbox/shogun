/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/io/openml/OpenMLFile.h>
#include <shogun/io/openml/OpenMLFlow.h>
#include <shogun/io/openml/utils.h>

#include <rapidjson/document.h>

using namespace shogun;
using namespace shogun::openml_detail;
using namespace rapidjson;

/**
 *
 */
class ShogunToString : public AnyVisitor
{
public:
	explicit ShogunToString(){SG_SDEBUG("Debugging ShogunToString\n")};

	void on(bool* v) final
	{
		m_string_val = (*v ? "true" : "false");
	}
	void on(int32_t* v) final
	{
		m_string_val = std::to_string(*v);
	}
	void on(int64_t* v) final
	{
		m_string_val = std::to_string(*v);
	}
	void on(float* v) final
	{
		m_string_val = std::to_string(*v);
	}
	void on(double* v) final
	{
		m_string_val = std::to_string(*v);
	}
	void on(long double* v)
	{
		m_string_val = std::to_string(*v);
	}
	void on(CSGObject** v) final
	{
		m_string_val = "";
	}
	void on(SGVector<int>* v) final
	{
		m_string_val = v->to_string();
	}
	void on(SGVector<float>* v) final
	{
		m_string_val = v->to_string();
	}
	void on(SGVector<double>* v) final
	{
		m_string_val = v->to_string();
	}
	void on(SGMatrix<int>* mat) final
	{
		m_string_val = mat->to_string();
	}
	void on(SGMatrix<float>* mat) final
	{
		m_string_val = mat->to_string();
	}
	void on(SGMatrix<double>* mat) final
	{
		m_string_val = mat->to_string();
	}

	SG_FORCED_INLINE std::string get_string_value() const noexcept
	{
		return m_string_val;
	}

private:
	std::string m_string_val;
};

std::shared_ptr<OpenMLFlow> OpenMLFlow::download_flow(
    const std::string& flow_id, const std::string& api_key)
{
	parameters_type params;
	components_type components;

	// get flow and parse with RapidJSON
	auto reader = OpenMLFile(api_key);
	auto return_string = reader.get("flow_file", "json", flow_id);

	auto& root = check_response<BACKEND_FORMAT::JSON>(return_string, "flow");

	std::string name =
	    return_if_possible<std::string>("name", root.GetObject());
	std::string description =
	    return_if_possible<std::string>("description", root.GetObject());
	std::string class_name =
	    return_if_possible<std::string>("class_name", root.GetObject());
	std::string external_version =
	    return_if_possible<std::string>("external_version", root.GetObject());

	REQUIRE(
	    root["id"].GetString() == flow_id,
	    "The flow id returned by the server does not match the id provided. "
	    "Got %s instead of %s.\n",
	    root["id"].GetString(), flow_id.c_str())

	// handle parameters
	if (root.HasMember("parameter"))
	{
		std::unordered_map<std::string, std::string> param_dict;
		OpenMLFlowParameter params_i{};

		if (root["parameter"].IsArray())
		{
			for (const auto& v : root["parameter"].GetArray())
			{
				add_string_to_struct(v, "name", params_i.name);
				add_string_to_struct(v, "data_type", params_i.data_type);
				add_string_to_struct(
				    v, "default_value", params_i.default_value);
				add_string_to_struct(v, "description", params_i.description);
				params.emplace(params_i.name, params_i);
			}
		}
		else
		{
			// parameter can also be a dict, instead of array
			const auto& v = root["parameter"].GetObject();
			add_string_to_struct(v, "name", params_i.name);
			add_string_to_struct(v, "data_type", params_i.data_type);
			add_string_to_struct(v, "default_value", params_i.default_value);
			add_string_to_struct(v, "description", params_i.description);
			params.emplace(params_i.name, params_i);
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
				    must_return<std::string>("identifier", v),
				    OpenMLFlow::download_flow(
				        v["flow"]["id"].GetString(), api_key));
			}
		}
		else
		{
			components.emplace(
			    must_return<std::string>("identifier", root["component"]),
			    OpenMLFlow::download_flow(
			        root["component"]["flow"]["id"].GetString(), api_key));
		}
	}

	auto flow = std::make_shared<OpenMLFlow>(
	    flow_id, name, description, class_name, external_version, components,
	    params);

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

bool OpenMLFlow::exists_on_server()
{
	// check if flow with same name and version exists
	auto reader = std::make_unique<OpenMLFile>("");
	auto flow_exists_string =
	    reader->get("flow_exists", "json", m_name, m_external_version);

	auto& root =
	    check_response<BACKEND_FORMAT::JSON>(flow_exists_string, "flow_exists");

	return strcmp(root["exists"].GetString(), "true") == 0;
}

std::vector<std::shared_ptr<OpenMLParameterValues>>
OpenMLFlow::obtain_parameter_values(const std::shared_ptr<CSGObject>& model)
{
	std::vector<std::shared_ptr<OpenMLParameterValues>> result;
	auto obj_param = model->get_params();
	auto visitor = std::make_unique<ShogunToString>();

	result.reserve(m_parameters.size());

	for (const auto& param : m_parameters)
	{
		Any any_val = obj_param.at(param.first)->get_value();
		any_val.visit(visitor.get());
		// nested objects are handled below
		if (!visitor->get_string_value().empty())
		{
			// result.emplace_back to call OpenMLParameterValues constructor
			// doesn't work here, so create a temporary value with make_shared
			// and then push_back
			auto val = std::make_shared<OpenMLParameterValues>(
			    param.first, m_flow_id, visitor->get_string_value());
			result.push_back(val);
		}
	}

	for (const auto& components : m_components)
	{
		// TODO: remove std::shared_ptr<CSGObject> when smart pointers available
		auto obj = std::shared_ptr<CSGObject>(model->get(components.first));
		auto val = std::make_shared<OpenMLParameterValues>(
		    components.first, m_flow_id,
		    components.second->obtain_parameter_values(obj));
		result.push_back(val);
	}

	return result;
}
