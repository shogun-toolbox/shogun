#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>
#include <shogun/mathematics/graph/operator_list.h>

#include <unordered_set>

using namespace shogun;

Graph::Graph(
    const std::vector<std::shared_ptr<Input>>& inputs,
    const std::vector<std::shared_ptr<Node>>& outputs)
{
	m_inputs = inputs;
	m_outputs = outputs;
}

void Graph::build()
{
	auto unordered_nodes = check_fully_connected(m_inputs, m_outputs);
	build_backend_graph(unordered_nodes);
}

std::unordered_map<std::shared_ptr<Node>, Graph::STATUS> Graph::check_fully_connected(
    const std::vector<std::shared_ptr<Input>>& inputs,
    const std::vector<std::shared_ptr<Node>>& outputs)
{
	std::deque<std::shared_ptr<Node>> nodes_to_check(outputs.begin(), outputs.end());
	std::unordered_set<std::shared_ptr<Node>> inputs_found;
	std::unordered_map<std::shared_ptr<Node>, STATUS> unordered_nodes;

	while (!nodes_to_check.empty())
	{
		auto top_node = nodes_to_check.front();
		nodes_to_check.pop_front();

		const auto& node_inputs = top_node->get_input_nodes();

		if (!node_inputs.empty())
		{
			for (const auto& node : node_inputs)
				nodes_to_check.push_back(node);
		}
		else if (
		    std::find(inputs.begin(), inputs.end(), top_node) != inputs.end())
		{
			// valid input node
			inputs_found.insert(top_node);
		}
		else
		{
			// has no children and is not an expected input
			error("Graph is disconnected in node {}.", *top_node);
		}
		unordered_nodes.emplace(top_node, STATUS::UNMARKED);
	}

	if (inputs_found.size() != inputs.size())
	{
		error("Graph found more input tensors than provided.");
	}

	m_cached_input_nodes = inputs;
	m_cached_output_nodes = outputs;

	return unordered_nodes;
}

std::vector<std::shared_ptr<Tensor>> Graph::evaluate(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
	if (m_cached_operators.empty())
	{
		error("Did you call Graph::build()?");
	}

	if (m_cached_input_operators.empty())
	{
		error("No input nodes found in graph!");
	}

	auto* env = ShogunEnv::instance();
	switch (env->graph_backend())
	{
	case GRAPH::SHOGUN:
	{
		execute_shogun(tensors);
	}
	break;
	case GRAPH::NGRAPH:
	{
#ifdef USE_NGRAPH
		execute_ngraph(tensors);
#else
		error("NGraph execution is not available.");
#endif
	}
	}

	std::vector<std::shared_ptr<Tensor>> result;

	for (const auto& node: m_cached_output_nodes)
	{
		result.push_back(node->get_tensors()[0]);
	}

	return result;
}


void Graph::build_backend_graph(
	std::unordered_map<std::shared_ptr<Node>, Graph::STATUS>& unordered_nodes)
{
	std::deque<std::shared_ptr<Node>> ordered_nodes;

	// DAG topological sorting algorithm with DFS
	for (const auto& node: unordered_nodes)
	{
		order_graph_visit_(node.first, unordered_nodes, ordered_nodes);
	}

	// get input operatos
	for (const auto& node: m_cached_input_nodes)
	{
		m_cached_input_operators.push_back(add_operator_node(node));
	}

	for (const auto& node: ordered_nodes)
	{
		// node not an input so safe to assume it's an operator
		if (std::find(m_cached_input_nodes.begin(), m_cached_input_nodes.end(), node) == m_cached_input_nodes.end())
			m_cached_operators.push_back(add_operator_node(node));
	}
}

void Graph::order_graph_visit_(const std::shared_ptr<Node>& node, 
	std::unordered_map<std::shared_ptr<Node>, Graph::STATUS>& all_nodes,
	std::deque<std::shared_ptr<Node>>& result)
{
	auto& node_status = all_nodes[node];
	if (node_status == STATUS::MARKED)
		return;
	if (node_status == STATUS::TEMPORARY)
		error("Not a DAG!");

	node_status = STATUS::TEMPORARY;

	for (auto& child_node: node->get_input_nodes())
	{
		order_graph_visit_(child_node, all_nodes, result);
	}

	node_status = STATUS::MARKED;
	result.push_back(node);
}

std::shared_ptr<Operator> Graph::add_operator_node(const std::shared_ptr<Node>& node)
{
	auto* env = ShogunEnv::instance();
	std::shared_ptr<Operator> op;

	switch (env->graph_backend())
	{
	case GRAPH::NGRAPH:
	{
#ifdef USE_NGRAPH
		
#endif
	}
	break;
	case GRAPH::XLA:
	case GRAPH::TVM:
	case GRAPH::SHOGUN:
		op = create_operator<OperatorShogunBackend>(std::string(node->get_operator_name()));
	}
	op->build(node);

	return op;
}


void Graph::execute_shogun(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
	if (tensors.size() != m_cached_input_operators.size())
		error("Number of input tensors ({}) different from number of input nodes ({}).", 
			tensors.size(), m_cached_input_nodes.size());
	for (const auto& [tensor, node]: zip_iterator(tensors, m_cached_input_operators))
	{
		std::static_pointer_cast<InputShogun>(node)->evaluate_input(tensor);
	}

	for (auto& op: m_cached_operators)
	{
		(*op)();
	}
}

void Graph::execute_ngraph(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
	// auto backend = ngraph::runtime::Backend::create("CPU", true);

	// std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	//     ngraph_input_tensors;
	// std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	//     ngraph_output_tensors;

	// for (const auto& tensor : input_tensors)
	// 	ngraph_input_tensors.push_back(backend->create_tensor(
	// 	    ngraph::element::f32, tensor.get_shape()));
	// for (const auto& tensor : output_tensors)
	// 	ngraph_output_tensors.push_back(backend->create_tensor(
	// 	    ngraph::element::f32, tensor.get_shape()));

	// auto handle = backend->compile(graph->get_ngraph_function());
	// handle->call_with_validate(
	//     ngraph_input_tensors, ngraph_output_tensors);

	// std::vector<Tensor> results;
	// for (const auto& ngraph_tensor : ngraph_output_tensors)
	// {
	// 	results.push_back(Tensor::create_empty(
	// 	    ngraph_tensor->get_shape(),
	// 	    get_enum_from_ngraph(ngraph_tensor->get_element_type())));
	// 	ngraph_tensor->read(
	// 	    results.back().data(),
	// 	    results.back().get_size() * sizeof(float));
	// }

	// return results;
}
