/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/util/factory.h>

#include <shogun/io/openml/ShogunOpenML.h>

using namespace shogun;

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

std::shared_ptr<CLabels> ShogunOpenML::run_model_on_fold(
		const std::shared_ptr<CSGObject>& model,
		const std::shared_ptr<OpenMLTask>& task,
		const std::shared_ptr<CFeatures>& X_train, index_t repeat_number,
		index_t fold_number, const std::shared_ptr<CLabels>& y_train,
		const std::shared_ptr<CFeatures>& X_test)
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
				// TODO: refactor! more useless clones until smart pointers are merged
				machine->put("labels", y_train->clone()->as<CLabels>());
				auto tmp = X_train.get();
				machine->train(tmp);
				if (X_test)
					return std::shared_ptr<CLabels>(machine->apply(X_test.get()));
				else
					return std::shared_ptr<CLabels>(machine->apply(X_train.get()));
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