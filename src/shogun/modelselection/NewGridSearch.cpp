/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/modelselection/NewGridSearch.h>

using namespace shogun;

const std::string ParameterNode::delimiter = "::";

ParameterNode::ParameterNode(CSGObject& model)
{
	m_parent = std::shared_ptr<CSGObject>(model.clone());
	for (auto const& param : model.get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			create_node(param.first, value);
			SG_UNREF(value);
		}
	}
}

void ParameterNode::create_node(const std::string& name, CSGObject* obj)
{
	SG_SPRINT("ParameterNode::create_node\n")
	m_nodes.insert(std::make_pair(
	    name, std::make_shared<ParameterNode>(ParameterNode(*obj))));
}

ParameterNode::ParameterNode(
    CSGObject& model, ParameterProperties param_properties)
{
	m_parent = std::shared_ptr<CSGObject>(model.clone());
	for (auto const& param : model.get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			m_nodes.insert(std::make_pair(
			    param.first,
			    std::make_shared<ParameterNode>(ParameterNode(*value))));
			SG_UNREF(value);
		}
	}

	for (auto const& param : model.get_params(param_properties))
	{
		m_param_mapping.insert(
		    std::make_pair(param.first, param.second->get_value()));
	}
}

ParameterNode* ParameterNode::attach(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	if (!set_param_helper(param, node))
		SG_SERROR("Could not attach %s", param.c_str());
	return this;
}

bool ParameterNode::set_param_helper(
    const std::string& param, const std::shared_ptr<ParameterNode>& node)
{
	auto param_iter = param.find(delimiter);

	if (param_iter != std::string::npos)
	{
		return set_param_helper(
		    param.substr(param_iter + delimiter.size()), node);
	}
	else
	{
		for (auto& node_i : m_nodes)
			if (node_i.first == param)
			{
				node_i.second = node;
				return true;
			}
	}

	return false;
}

std::string ParameterNode::to_string() const
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
		ss << ", ";

		for (auto node = m_nodes.begin(); node != m_nodes.end(); ++node)
		{
			ss << (*node).first << "=";
			(*node).second->to_string_helper(ss, visitor);
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

	if (param_iter != std::string::npos)
	{
		for (auto& node : m_nodes)
		{
			return node.second->set_param_helper(
			    param.substr(param_iter + delimiter.size()), value);
		}
	}
	else
	{
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
	auto tree = new ParameterNode(*m_parent);

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

CSGObject* ParameterNode::to_object(const std::shared_ptr<ParameterNode>& tree)
{
}

GridParameters::GridParameters(CSGObject& model)
    : first(true), m_node_complete(false)
{
	m_parent = std::shared_ptr<CSGObject>(model.clone());
	// TODO: once all hyperparameters are flagged properly add
	//  ParameterProperties::HYPER
	for (auto const& param : model.get_params())
	{
		if (auto* value = m_parent->get(param.first, std::nothrow))
		{
			create_node(param.first, value);
			SG_UNREF(value);
		}
	}
	m_current_node = m_nodes.begin();
}

void GridParameters::reset()
{
	for (const auto& param_pair : m_param_mapping)
		m_param_iter[param_pair.first] = m_param_begin[param_pair.first];
	m_node_complete = true;
	m_current_param = m_param_mapping.end();
	first = true;
}

void GridParameters::create_node(const std::string& name, CSGObject* obj)
{
	m_nodes.insert(std::make_pair(
	    name, std::make_shared<GridParameters>(GridParameters(*obj))));
}

ParameterNode* GridParameters::get_next()
{
	SG_SPRINT("GridParameters::get_next() - %s\n", get_parent_name());
	auto tree = new ParameterNode(*m_parent);

	std::string param;

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

	auto increment_lambda = [&param, this](auto val) {
		auto beginning = any_cast<decltype(val.begin())>(m_param_begin[param]);
		auto current = any_cast<decltype(val.begin())>(m_param_iter[param]);
		auto end = any_cast<decltype(val.begin())>(m_param_end[param]);

		if (std::next(current) != end &&
		    m_current_param != m_param_mapping.end())
		{
			SG_SPRINT("not the end and not the last param: %s\n", param.c_str())
			m_param_iter[param] = make_any(++beginning);
			m_current_param = m_param_mapping.end();
		}
		else if (std::next(current) == end)
		{
			// this is the beginning so need to go back to end of map
			if (m_current_param == m_param_mapping.begin())
			{
				m_param_iter[param] = make_any(beginning);
				SG_SPRINT("the end and the last param: %s\n", param.c_str())
				m_current_param = m_param_mapping.end();
				SG_SPRINT(
				    "updating m_current_param from %s::%s to %s::%s\n",
				    get_parent_name(), param.c_str(), get_parent_name(),
				    m_current_param->first.c_str())
			}
			// find the next param to iterate over
			else
			{
				bool found_next_param = false;
				auto increment_neighbour = [&param, &found_next_param,
				                            this](auto val) {
					auto beginning =
					    any_cast<decltype(val.begin())>(m_param_begin[param]);
					auto current =
					    any_cast<decltype(val.begin())>(m_param_iter[param]);
					auto end =
					    any_cast<decltype(val.begin())>(m_param_end[param]);

					if (std::next(current) == end)
					{
						SG_SPRINT(
						    "increment_neighbour - end of iteration param: "
						    "%s\n",
						    param.c_str())
						m_param_iter[param] = make_any(beginning);

						if (std::prev(m_current_param) == m_param_mapping.begin())
						{
							SG_SPRINT(
							    "the end and the last param: %s\n",
							    param.c_str())
							m_current_param = m_param_mapping.end();
							SG_SPRINT(
							    "updating m_current_param from %s::%s to "
							    "%s::%s\n",
							    get_parent_name(), param.c_str(),
							    get_parent_name(),
							    m_current_param->first.c_str())
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
						SG_SPRINT(
						    "increment_neighbour - not the end of iteration "
						    "param: %s\n",
						    param.c_str())
						m_param_iter[param] = make_any(++current);
						found_next_param = true;
					}
				};
				SG_SPRINT("the end and not the last param: %s\n", param.c_str())
				m_current_param = m_param_mapping.end();
				param = std::prev(m_current_param)->first;
				while (!found_next_param)
				{
					sg_any_dispatch(
					    m_param_mapping[param], sg_vector_typemap,
					    shogun::None{}, increment_neighbour);
				}
				SG_SPRINT(
				    "node is done updating m_current_param to "
				    "%s::%s\n",
				    get_parent_name(), param.c_str(), get_parent_name(),
				    std::prev(m_current_param)->first.c_str())
			}
		}
		else
		{
			SG_SPRINT("not the end: %s\n", param.c_str())
			m_param_iter[param] = make_any(++current);
		}
	};

	for (const auto& node : m_nodes)
	{
		if (*m_current_node == node)
			tree->attach(
			    node.first,
			    std::shared_ptr<ParameterNode>(
			        std::dynamic_pointer_cast<GridParameters>(node.second)
			            ->get_next()));
		else
			tree->attach(
			    node.first,
			    std::shared_ptr<ParameterNode>(
			        std::dynamic_pointer_cast<GridParameters>(node.second)
			            ->get_current()));
	}

	if (m_current_node != m_nodes.end() && !m_nodes.empty())
	{
		SG_SPRINT(
		    "DEBUG CHECK NODE IS DONE: %s\n",
		    m_current_node->second->get_parent_name())
		if (std::dynamic_pointer_cast<GridParameters>(m_current_node->second)
		        ->check_child_node_done())
		{
			SG_SPRINT(
			    "DEBUG NODE IS DONE: %s\n",
			    m_current_node->second->get_parent_name())
			++m_current_node;
		}

		if (m_current_node == m_nodes.end())
			m_current_node = m_nodes.begin();
	}

	if (first)
	{
		m_current_param = m_param_mapping.end();
		first = false;
	}

	for (auto param_pair = m_param_mapping.begin();
	     param_pair != m_param_mapping.end(); ++param_pair)
	{
		param = param_pair->first;

		SG_SPRINT(
		    "GridParameters::get_next - param: %s, current_param: %s\n",
		    param.c_str(), m_current_param->first.c_str())

		if (m_nodes.empty())
		{
			// check if need to iterate bottom node param
			if (std::next(param_pair) == m_current_param)
			{
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    increment_lambda);
				for (; param_pair != m_param_mapping.end(); ++param_pair)
				{
					param = param_pair->first;
					sg_any_dispatch(
					    param_pair->second, sg_vector_typemap, shogun::None{},
					    get_lambda);
				}
				if (m_node_complete)
					reset();
				return tree;
			}
			else
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
		}
		else
		{
			// check if need to iterate params of this node or child nodes
			if (std::dynamic_pointer_cast<GridParameters>(
			        m_current_node->second)
			        ->check_child_node_done() &&
			    (param_pair == m_current_param ||
			     std::next(param_pair) == m_param_mapping.end()))
			{
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    increment_lambda);
				std::dynamic_pointer_cast<GridParameters>(
				    m_current_node->second)
				    ->set_node_complete(false);
			}
			else
				sg_any_dispatch(
				    param_pair->second, sg_vector_typemap, shogun::None{},
				    get_lambda);
		}
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
			return std::dynamic_pointer_cast<GridParameters>(node.second)
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
			if (!std::dynamic_pointer_cast<GridParameters>(node.second)
			         ->check_child_node_done())
				return false;
		}
	}
	else
		return false;
	return true;
}

ParameterNode* GridParameters::get_current()
{
	auto tree = new ParameterNode(*m_parent);

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
