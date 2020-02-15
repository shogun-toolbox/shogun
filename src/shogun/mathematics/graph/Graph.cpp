#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/Tensor.h>

#include <memory>

#ifdef USE_NGRAPH
#include <ngraph/ngraph.h>
#endif

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
	if (!m_executor)
	{
		error("Graph has not been built!");
	}

	m_executor->execute(tensors);

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

	auto* env = ShogunEnv::instance();
	m_executor = create(env->graph_backend());
	if (!m_executor)
		error("Specified graph executor {} backend is not available!",
			kGraphNames.at(env->graph_backend()));

	// get input operatos
	for (const auto& node: m_cached_input_nodes)
	{
		m_executor->add_input_operator(node);
	}

	for (const auto& node: ordered_nodes)
	{
		// node not an input so safe to assume it's an operator
		if (std::find(m_cached_input_nodes.begin(), m_cached_input_nodes.end(), node) == m_cached_input_nodes.end())
			m_executor->add_operator_node(node);
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
