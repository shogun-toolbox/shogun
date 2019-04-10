/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/distance/Distance.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/modelselection/NewGridSearch.h>
#include <shogun/util/factory.h>

using namespace shogun;

const std::string ParameterNode::delimiter = "::";

ParameterNode::ParameterNode(CSGObject* model)
{
	SG_REF(model)
	m_parent = model;
	SG_SDEBUG("%s refcount: %d\n", model->get_name(), m_parent->ref_count())
	for (auto const& param : m_parent->get_params())
	{
		// check the param is neither of type CLabels or CFeatures
		if ((std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CLabels*))) &&
		    (std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CFeatures*))))
		{
			if (auto* value = m_parent->get(param.first, std::nothrow))
			{
				auto node = new ParameterNode(value);
				create_node(param.first, node);
			}
		}
	}
}

void ParameterNode::create_node(const std::string& name, ParameterNode* obj)
{
	m_nodes[name].emplace_back(obj);
}

ParameterNode::ParameterNode(
    CSGObject* model, ParameterProperties param_properties)
{
	SG_REF(model)
	m_parent = model;

	for (auto const& param : m_parent->get_params())
	{
		// check the param is neither of type CLabels or CFeatures
		if ((std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CLabels*))) &&
		    (std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CFeatures*))))
		{
			if (auto* value = m_parent->get(param.first, std::nothrow))
			{
				auto node = new ParameterNode(value);
				create_node(param.first, node);
			}
		}
	}

	for (auto const& param : model->get_params(param_properties))
	{
		m_param_mapping.insert(
		    std::make_pair(param.first, param.second->get_value()));
	}
}

ParameterNode* ParameterNode::attach(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	return attach(param, node.get());
}

ParameterNode*
ParameterNode::attach(const std::string& param, ParameterNode* node)
{
	auto* node_copy = new ParameterNode(*node);
	if (!set_param_helper(param, node_copy))
		SG_SERROR("Could not attach %s\n", param.c_str());
	return this;
}

ParameterNode*
ParameterNode::attach(const std::string& param, ParameterNode&& node)
{
	if (!set_param_helper(param, &node))
		SG_SERROR("Could not attach %s\n", param.c_str());
	return this;
}

bool ParameterNode::set_param_helper(
    const std::string& param, ParameterNode* node)
{
	auto param_iter = param.find(delimiter);

	if (param_iter != std::string::npos)
	{
		return set_param_helper(
		    param.substr(param_iter + delimiter.size()), node);
	}
	else
	{
		if (m_parent->has(param))
		{
			create_node(param, node);
			return true;
		}
	}
	return false;
}

std::string ParameterNode::to_string() const noexcept
{
	std::stringstream ss;
	std::unique_ptr<AnyVisitor> visitor(new ToStringVisitor(&ss));
	ss << get_name() << "(" << get_parent_name() << "(";

	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		ss << param_pair->first.c_str() << "=";
		param_pair->second.visit(visitor.get());
		if (std::next(param_pair) != m_param_mapping.end())
			ss << ", ";
	}

	if (!m_nodes.empty())
	{
		if (!m_param_mapping.empty())
			ss << ", ";

		for (auto node = m_nodes.begin(); node != m_nodes.end(); ++node)
		{
			ss << (*node).first << "=";
			if (node->second.size() > 1)
			{
				ss << "[";
			}
			for (auto inner_node = node->second.begin();
			     inner_node != node->second.end(); ++inner_node)
			{
				(*inner_node)->to_string_helper(ss, visitor);
				if (std::next(inner_node) != node->second.end())
					ss << ", ";
			}
			if (node->second.size() > 1)
			{
				ss << "]";
			}
			if (std::next(node) != m_nodes.end())
				ss << ", ";
		}
	}
	ss << "))";
	return ss.str();
}

void ParameterNode::to_string_helper(
    std::stringstream& ss, std::unique_ptr<AnyVisitor>& visitor) const noexcept
{
	ss << get_parent_name() << "(";
	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		ss << param_pair->first.c_str() << "=";
		param_pair->second.visit(visitor.get());
		if (std::next(param_pair) != m_param_mapping.end())
			ss << ", ";
	}
	ss << ")";
}

bool ParameterNode::set_param_helper(const std::string& param, const Any& value)
{
	auto param_iter = param.find(delimiter);
	SG_SDEBUG("ParameterNode::set_param_helper - param: %s\n", param.c_str())

	if (param_iter != std::string::npos)
	{
		SG_SDEBUG(
		    "ParameterNode::set_param_helper - Looking for param in nodes: "
		    "%s\n",
		    param.c_str())
		for (auto& node : m_nodes)
		{
			if (node.second.size() > 1)
			{
				SG_SERROR(
				    "Ambiguous call! More than one %s found in "
				    "%s::%s. To specify a parameter in %s use "
				    "the attach method\n",
				    get_name(), get_parent_name(), param.c_str(), get_name())
				return false;
			}
			else
			{
				SG_SDEBUG(
				    "ParameterNode::set_param_helper - Looking for %s in %s\n",
				    param.substr(param_iter + delimiter.size()).c_str(),
				    node.first.c_str())
				return node.second[0]->set_param_helper(
				    param.substr(param_iter + delimiter.size()), value);
			}
		}
	}
	else
	{
		SG_SDEBUG(
		    "ParameterNode::set_param_helper - Looking for %s in %s\n",
		    param.c_str(), m_parent->get_name())
		if (m_parent->has(param))
		{
			m_param_mapping.insert(std::make_pair(param, value));
			return true;
		}
	}
	return false;
}

ParameterNode* ParameterNode::get_current()
{
	auto tree = new ParameterNode(m_parent);

	std::string param;

	auto get_lambda = [&tree, &param](auto val) { tree->attach(param, val); };

	for (const auto& param_pair : m_param_mapping)
	{
		param = param_pair.first;
		sg_any_dispatch(
		    param_pair.second, sg_vector_typemap, shogun::None{}, get_lambda);
	}
	return tree;
}

CSGObject* ParameterNode::to_object(const std::unique_ptr<ParameterNode>& tree)
{
	SG_SDEBUG("CSGObject* ParameterNode::to_object(const "
	          "std::unique_ptr<ParameterNode>& tree)\n")
	SG_SDEBUG(
	    "REFCOUNT %s %d\n", tree->m_parent->get_name(),
	    tree->m_parent->ref_count())
	auto* result = tree->m_parent->clone();
	SG_SDEBUG("%s: %d\n", result->get_name(), result->ref_count())
	auto result_params = result->get_params();
	for (const auto& node : tree->m_nodes)
	{
		auto type_index_i =
		    std::type_index(result_params[node.first]->get_value().type_info());
		auto* obj = ParameterNode::to_object(node.second[0]);
		if (type_index_i == std::type_index(typeid(CKernel*)))
		{
			SG_SDEBUG(
			    "REFCOUNT - obj %s %d\n", obj->get_name(), obj->ref_count())
			result->put(node.first, obj->as<CKernel>());
		}
		else if (type_index_i == std::type_index(typeid(CDistance*)))
		{
			SG_SDEBUG(
			    "REFCOUNT - obj %s %d\n", obj->get_name(), obj->ref_count())
			result->put(node.first, obj->as<CDistance>());
		}
		else
		{
			SG_SERROR(
			    "Unsupported type %s for parameter %s::%s\n",
			    demangled_type(type_index_i.name()).c_str(), result->get_name(),
			    obj->get_name());
		}
		// obj is not needed in this scope anymore
		SG_UNREF(obj)
		SG_SDEBUG("REFCOUNT - obj %s %d\n", obj->get_name(), obj->ref_count())
	}
	std::string param_name;
	auto put_scalar_lambda = [&result, &param_name](const auto& val) {
		result->put(param_name, val);
	};
	for (const auto& param : tree->m_param_mapping)
	{
		param_name = param.first;
		sg_any_dispatch(param.second, sg_all_typemap, put_scalar_lambda);
	}
	return result;
}

void ParameterNode::replace_node(
    const std::string& node_name, size_t index, ParameterNode* node)
{
	auto it = m_nodes.find(node_name);

	if (it != m_nodes.end())
	{
		if (index < (*it).second.size())
			(*it).second[index].reset(node);
		else
		{
			SG_SERROR(
			    "Index out of range for %s::%s", get_parent_name(),
			    node_name.c_str())
		}
	}
	else
	{
		SG_SERROR(
		    "Unable to find and replace %s::%s", get_parent_name(),
		    node_name.c_str())
	}
}

ParameterNode::~ParameterNode(){
    SG_SDEBUG("~%s\n", get_name()) SG_SDEBUG(
        "REFCOUNT %s %d\n", m_parent->get_name(), m_parent->ref_count())
        SG_UNREF(m_parent) SG_SDEBUG(
            "REFCOUNT %s %d\n", m_parent->get_name(), m_parent->ref_count())}

ParameterNode::ParameterNode(const ParameterNode& other)
{
	SG_REF(other.m_parent)
	m_parent = other.m_parent;
	SG_SDEBUG("%s refcount: %d\n", m_parent->get_name(), m_parent->ref_count())
	for (const auto& node : other.m_nodes)
	{
		for (const auto& inner_node : node.second)
			create_node(node.first, inner_node.get());
	}

	for (const auto& param : other.m_param_mapping)
	{
		m_param_mapping[param.first].clone_from(
		    other.m_param_mapping.at(param.first));
	}
}

GridParameters::GridParameters(CSGObject* model)
    : m_first(true), m_node_complete(false)
{
	SG_REF(model)
	m_parent = model;
	SG_SDEBUG("%s refcount: %d\n", m_parent->get_name(), m_parent->ref_count())
	// TODO: once all hyperparameters are flagged properly replace getter with
	//  get_params.model(ParameterProperties::HYPER)
	for (auto const& param : model->get_params())
	{
		// check the param is neither of type CLabels or CFeatures
		if ((std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CLabels*))) &&
		    (std::type_index(param.second->get_value().type_info()) !=
		     std::type_index(typeid(CFeatures*))))
		{
			if (auto* value = m_parent->get(param.first, std::nothrow))
			{
				auto node = new GridParameters(value);
				create_node(param.first, node);
			}
		}
	}
	m_current_node = m_nodes.begin();
	m_current_internal_node = m_current_node->second.begin();
}

GridParameters::GridParameters(const GridParameters& other)
    : ParameterNode(other), m_first(other.m_first),
      m_current_node(other.m_current_node),
      m_current_internal_node(other.m_current_internal_node),
      m_current_param(other.m_current_param), m_param_iter(other.m_param_iter),
      m_param_begin(other.m_param_begin), m_param_end(other.m_param_end),
      m_node_complete(other.m_node_complete)
{
}

void GridParameters::reset()
{
	for (const auto& param_pair : m_param_mapping)
		m_param_iter[param_pair.first] = m_param_begin[param_pair.first];
	m_node_complete = false;
	m_current_param = m_param_mapping.end();
	m_first = true;
}

ParameterNode* GridParameters::get_next()
{
	if (is_complete())
	{
		reset();
	}
	if (m_current_node != m_nodes.end() && !m_nodes.empty())
	{
		// iterate over the parameter of this node so can set
		// m_current_internal_node to false
		SG_SDEBUG(
		    "GridParameters::get_next - DEBUG CHECK NODE IS DONE: %s\n",
		    (*m_current_internal_node)->get_parent_name())
		bool node_is_done =
		    dynamic_cast<GridParameters*>((*m_current_internal_node).get())
		        ->check_child_node_done();
		if (node_is_done)
		{
			SG_SDEBUG(
			    "GridParameters::get_next - DEBUG NODE IS DONE: %s\n",
			    (*m_current_internal_node)->get_parent_name())
			++m_current_internal_node;
		}

		if (m_current_internal_node == m_current_node->second.end())
		{
			if (std::next(m_current_node) != m_nodes.end())
				++m_current_node;
			else
				m_current_node = m_nodes.begin();
			m_current_internal_node = m_current_node->second.begin();
			SG_SDEBUG(
			    "GridParameters::get_next - CURRENT NODE IS: %s \n",
			    (*m_current_internal_node)->get_parent_name())
		}
	}

	auto tree = new ParameterNode(m_parent);

	std::string param;

	// Lambda responsible with getting the current value.
	// Also sets m_current_param to the last param in map
	auto get_lambda = [&tree, &param, this](auto val) {
		// decltype(val.begin()) is a "hacky" way of getting the iterator type
		// without registering RandomIterator<T> in type_case.h but need to pass
		// SGVector by value each time
		auto current = any_cast<decltype(val.begin())>(m_param_iter[param]);

		tree->attach(param, *current);

		// if this is not the end node we need to go back to it after fetching
		// the current value
		if (m_current_param != m_param_mapping.end())
		{
			m_current_param = m_param_mapping.end();
		}
	};

	// Lambda responsible with iterating and finding the next parameter to
	// iterate over. Also signals the end of a node
	auto increment_lambda = [&param, this](auto val) {
		auto beginning = any_cast<decltype(val.begin())>(m_param_begin[param]);
		auto current = any_cast<decltype(val.begin())>(m_param_iter[param]);
		auto end = any_cast<decltype(val.begin())>(m_param_end[param]);

		if (std::next(current) != end &&
		    m_current_param != m_param_mapping.end())
		{
			SG_SDEBUG(
			    "GridParameters::get_next - not the end and not the last "
			    "param: %s\n",
			    param.c_str())
			m_param_iter[param] = make_any(++beginning);
			m_current_param = m_param_mapping.end();
		}
		else if (std::next(current) == end)
		{
			// this is the beginning so need to go back to end of map
			if (m_current_param == m_param_mapping.begin())
			{
				m_param_iter[param] = make_any(beginning);
				SG_SDEBUG(
				    "GridParameters::get_next - the end and the last param: "
				    "%s\n",
				    param.c_str())
				m_current_param = m_param_mapping.end();
				SG_SDEBUG(
				    "GridParameters::get_next - "
				    "updating m_current_param from %s::%s to %s::%s\n",
				    this->get_parent_name(), param.c_str(),
				    this->get_parent_name(), m_current_param->first.c_str())
			}
			// find the next param to iterate over
			else
			{
				bool found_next_param = false;
				auto increment_neighbour = [&param, &found_next_param,
				                            this](auto val) {
					auto inner_beginning =
					    any_cast<decltype(val.begin())>(m_param_begin[param]);
					auto inner_current =
					    any_cast<decltype(val.begin())>(m_param_iter[param]);
					auto inner_end =
					    any_cast<decltype(val.begin())>(m_param_end[param]);

					if (std::next(inner_current) == inner_end)
					{
						SG_SDEBUG(
						    "GridParameters::get_next - "
						    "increment_neighbour - end of iteration param: "
						    "%s\n",
						    param.c_str())
						m_param_iter[param] = make_any(inner_beginning);

						if (std::prev(m_current_param) ==
						    m_param_mapping.begin())
						{
							SG_SDEBUG(
							    "GridParameters::get_next - "
							    "the end and the last param: %s\n",
							    param.c_str())
							m_current_param = m_param_mapping.end();
							SG_SDEBUG(
							    "GridParameters::get_next - "
							    "updating m_current_param from %s::%s to "
							    "%s::%s\n",
							    this->get_parent_name(), param.c_str(),
							    this->get_parent_name(),
							    std::prev(m_current_param)->first.c_str())
							found_next_param = true;
							m_node_complete = true;
						}
						else
						{
							found_next_param = false;
							m_current_param = std::prev(m_current_param);
							param = std::prev(m_current_param)->first;
						}
					}
					else
					{
						SG_SDEBUG(
						    "GridParameters::get_next - "
						    "increment_neighbour - not the end of iteration "
						    "param: %s\n",
						    param.c_str())
						m_param_iter[param] = make_any(++inner_current);
						found_next_param = true;
					}
				};
				SG_SDEBUG(
				    "GridParameters::get_next - "
				    "the end and not the last param: %s\n",
				    param.c_str())
				m_current_param = m_param_mapping.end();
				param = std::prev(m_current_param)->first;
				while (!found_next_param)
				{
					sg_any_dispatch(
					    m_param_mapping[param], sg_vector_typemap,
					    shogun::None{}, increment_neighbour);
				}
				SG_SDEBUG(
				    "GridParameters::get_next - "
				    "node is done updating m_current_param to "
				    "%s::%s\n",
				    this->get_parent_name(), param.c_str(),
				    this->get_parent_name(),
				    std::prev(m_current_param)->first.c_str())
			}
		}
		else
		{
			SG_SDEBUG(
			    "GridParameters::get_next - "
			    "not the end: %s\n",
			    param.c_str())
			m_param_iter[param] = make_any(++current);
		}
	};

	SG_SDEBUG(
	    "GridParameters::get_next - "
	    "FIRST 1: %s\n",
	    tree->to_string().c_str());

	for (const auto& node : m_nodes)
	{
		ParameterNode* this_node(nullptr);

		ParameterNode* (GridParameters::*fp)() = nullptr;

		if (node != *m_current_node)
			fp = &GridParameters::get_current;
		else
			fp = &GridParameters::get_next;

		for (auto inner_node = node.second.begin();
		     inner_node != node.second.end(); ++inner_node)
		{
			// replace the node of the object with current parameter
			// representation
			SG_SDEBUG(
			    "GridParameters::get_next - "
			    "INNER NODE %s\n",
			    inner_node->get()->get_parent_name())
			if (inner_node == m_current_internal_node)
			{
				this_node =
				    (*dynamic_cast<GridParameters*>((*inner_node).get()).*fp)();
				break;
			}
		}
		// need to check if this is an attach or replace, there are
		// situations where the tree isn't aware at this point that it
		// needs nodes, i.e. node is attached after calling constructor
		if (tree->n_nodes() > 0)
			tree->replace_node(node.first, 0, this_node);
		else
		{
			// getters return a copy so call attach that takes r val ref,
			// which doesn't copy obj
			tree->attach(node.first, std::move(*this_node));
		}
	}

	SG_SDEBUG(
	    "GridParameters::get_next - "
	    "FIRST 2: %s\n",
	    tree->to_string().c_str());

	if (m_first)
	{
		m_current_param = m_param_mapping.end();
		m_first = false;
	}

	bool iterate_this_loop = false;

	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		param = param_pair->first;

		SG_SDEBUG(
		    "GridParameters::get_next - param: %s, current_param: %s\n",
		    param.c_str(), std::prev(m_current_param)->first.c_str())

		if (m_nodes.empty())
		{
			// check if need to iterate bottom node param
			iterate_this_loop = std::next(param_pair) == m_current_param;
		}
		else
		{
			// check if need to iterate params of this node or child nodes
			iterate_this_loop =
			    dynamic_cast<GridParameters*>((*m_current_internal_node).get())
			        ->check_child_node_done() &&
			    std::next(m_current_internal_node) ==
			        m_current_node->second.end() &&
			    std::next(param_pair) == m_current_param;
		}

		if (iterate_this_loop)
		{
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    get_lambda);
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    increment_lambda);
			// we know that nothing else can be iterated so just use a nested
			// loop to avoid any further dynamic casting
			for (; param_pair != m_param_mapping.end(); ++param_pair)
			{
				param = param_pair->first;
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
			}
			break;
		}
		else
			sg_any_dispatch(
			    param_pair->second, sg_vector_typemap, shogun::None{},
			    get_lambda);
	}

	return tree;
}

bool GridParameters::set_param_helper(
    const std::string& param, const Any& value)
{
	auto param_iter = param.find(delimiter);

	auto set_iter_lambda = [param, &m_param_begin = m_param_begin,
	                        &m_param_iter = m_param_iter,
	                        &m_param_end = m_param_end](auto val) {
		m_param_begin.insert(std::make_pair(param, make_any(val.begin())));
		m_param_iter.insert(std::make_pair(param, make_any(val.begin())));
		m_param_end.insert(std::make_pair(param, make_any(val.end())));
	};

	if (param_iter != std::string::npos)
	{
		for (auto& node : m_nodes)
		{
			if (node.second.size() > 1)
			{
				SG_SERROR(
				    "Ambiguous call! More than one ParameterNode found in "
				    "%s::%s. To specify a parameter in this ParameterNode use "
				    "the attach method of ParameterNode %s",
				    get_parent_name(), param.c_str(), get_parent_name())
				return false;
			}
			else
				return dynamic_cast<GridParameters*>(node.second[0].get())
				    ->set_param_helper(
				        param.substr(param_iter + delimiter.size()), value);
		}
	}
	else
	{
		if (m_parent->has(param))
		{
			m_param_mapping.insert(std::make_pair(param, value));
			sg_any_dispatch(
			    value, sg_all_typemap, shogun::None{}, set_iter_lambda);
			return true;
		}
	}
	return false;
}

bool GridParameters::is_complete()
{
	if (check_child_node_done())
	{
		for (const auto& node : m_nodes)
		{
			if (!is_inner_complete(node.first))
				return false;
		}
	}
	else
		return false;
	return true;
}

bool GridParameters::is_inner_complete(const std::string& node_name)
{
	for (const auto& inner_node : m_nodes[node_name])
	{
		if (!dynamic_cast<GridParameters*>(inner_node.get())
		         ->check_child_node_done())
			return false;
	}
	return true;
}

ParameterNode* GridParameters::get_current()
{
	auto tree = new ParameterNode(m_parent);

	std::string param;

	auto get_lambda = [&tree, &param, &m_param_iter = m_param_iter](auto val) {
		tree->attach(
		    param, *any_cast<decltype(val.begin())>(m_param_iter[param]));
	};

	for (auto& param_pair : m_param_mapping)
	{
		param = param_pair.first;
		sg_any_dispatch(
		    param_pair.second, sg_vector_typemap, shogun::None{}, get_lambda);
	}
	return tree;
}

ParameterNode*
GridParameters::attach(const std::string& param, ParameterNode* node)
{
	auto* gp_node = dynamic_cast<GridParameters*>(node);
	if (!gp_node)
		SG_SERROR("Need to pass an object of type GridParameters")
	auto* param_copy = new GridParameters(*gp_node);
	if (!ParameterNode::set_param_helper(param, param_copy))
		SG_SERROR("Could not attach %s", param.c_str());
	// Updating node list (might) require resetting the node iterators
	m_current_node = m_nodes.begin();
	m_current_internal_node = m_current_node->second.begin();
	return this;
}

ParameterNode* GridParameters::attach(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	return attach(param, node.get());
}

CSGObject* GridSearch::train(CFeatures* features, CLabels* labels, bool verbose)
{
	auto parameter_combination = std::unique_ptr<ParameterNode>(get_next());
	SG_SDEBUG(
	    "GridSearch::train - "
	    "CURRENT NODE: %s\n",
	    parameter_combination->to_string().c_str())
	auto* machine =
	    ParameterNode::to_object(parameter_combination)->as<CMachine>();
	SG_SDEBUG("machine: %d\n", machine->ref_count())
	m_strategy->put("labels", labels);
	// TODO: add non lock option
	auto eval_machine = new CCrossValidation(
	    machine, features, labels, m_strategy, m_evaluation, false);
	SG_UNREF(machine)

	std::unique_ptr<ParameterNode> best_combination =
	    std::move(parameter_combination);
	auto* best_result = (CCrossValidationResult*)(eval_machine->evaluate());

	// apply all combinations and search for best one
	while (!is_complete())
	{
		// delete old parameter combination and replace with new one
		SG_SDEBUG("GridSearch::train - GETTING NEW COMBINATION\n")
		parameter_combination.reset(get_next());
		SG_SDEBUG("GridSearch::train - GOT NEW COMBINATION\n")

		// the machine is returned with a ref_count of 1 because of clone
		auto* tmp =
		    ParameterNode::to_object(parameter_combination)->as<CMachine>();

		// the old machine can now be unref'ed
		auto* old = eval_machine->get<CMachine*>("machine");
		SG_SDEBUG(
		    "old %s: %d\n", eval_machine->get<CMachine*>("machine")->get_name(),
		    eval_machine->get<CMachine*>("machine")->ref_count())
		SG_UNREF(old)

		eval_machine->put("machine", tmp);
		// tmp should only be known by eval_machine
		SG_UNREF(tmp)
		SG_SDEBUG("tmp %s: %d\n", tmp->get_name(), tmp->ref_count())

		if (verbose)
		{
			SG_SPRINT(
			    "GridSearch::train - "
			    "trying combination: %s\n",
			    parameter_combination->to_string().c_str())
			SG_SPRINT(
			    "GridSearch::train - "
			    "trying model: %s\n",
			    eval_machine->get<CMachine*>("machine")->to_string().c_str())
		}

		// note that this may implicitly lock and unlock the machine
		auto* result = (CCrossValidationResult*)(eval_machine->evaluate());

		if (result->get_result_type() != CROSSVALIDATION_RESULT)
			SG_SERROR(
			    "Evaluation result is not of type CCrossValidationResult!")

		if (verbose)
			result->print_result();

		auto lambda_compare =
		    [&eval_machine](const auto& lhs, const auto& rhs) {
			    if (eval_machine->get_evaluation_direction() == ED_MAXIMIZE)
				    return lhs > rhs;
			    return lhs < rhs;
		    };

		// check if current result is better, replace old combinations
		if (lambda_compare(result->get_mean(), best_result->get_mean()))
		{
			SG_SDEBUG("GridSearch::train - "
			          "REPLACING BEST COMBINATION")
			best_combination = std::move(parameter_combination);

			SG_REF(result);
			SG_UNREF(best_result);
			best_result = result;
		}
		SG_UNREF(result);
	}

	if (verbose)
	{
		SG_SPRINT(
		    "GridSearch::train - "
		    "Best result: %f, Best parameter combination: %s\n",
		    best_result->get_mean(), best_combination->to_string().c_str())
	}
	SG_UNREF(best_result)
	SG_UNREF(eval_machine)

	auto* best_model = ParameterNode::to_object(best_combination);
	// return the SGObject with refcount = 1
	return best_model;
}
