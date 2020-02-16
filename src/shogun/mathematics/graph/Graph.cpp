#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/Tensor.h>

#include <memory>

using namespace shogun::graph;
using namespace std;

Graph::Graph(
    const vector<shared_ptr<node::Input>>& inputs,
    const vector<shared_ptr<node::Node>>& outputs)
{
	m_inputs = inputs;
	m_outputs = outputs;
}

void Graph::build()
{
	auto* env = ShogunEnv::instance();
	build(env->graph_backend());
}

void Graph::build(GRAPH_BACKEND backend)
{
	auto unordered_nodes = check_fully_connected(m_inputs, m_outputs);

	m_executor = create(backend);
	if (!m_executor)
		error(
		    "Specified graph executor '{}' is not available!",
		    kGraphNames.at(backend));

	build_backend_graph(unordered_nodes);
}

unordered_map<shared_ptr<node::Node>, Graph::STATUS>
Graph::check_fully_connected(
    const vector<shared_ptr<node::Input>>& inputs,
    const vector<shared_ptr<node::Node>>& outputs)
{
	deque<shared_ptr<node::Node>> nodes_to_check(
	    outputs.begin(), outputs.end());
	unordered_set<shared_ptr<node::Node>> inputs_found;
	unordered_map<shared_ptr<node::Node>, STATUS> unordered_nodes;

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
		else if (find(inputs.begin(), inputs.end(), top_node) != inputs.end())
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

	return unordered_nodes;
}

vector<shared_ptr<Tensor>>
Graph::evaluate(const vector<shared_ptr<Tensor>>& tensors)
{
	if (!m_executor)
	{
		error("Graph has not been built!");
	}

	return m_executor->execute(tensors, m_outputs);
}

void Graph::build_backend_graph(
    unordered_map<shared_ptr<node::Node>, Graph::STATUS>& unordered_nodes)
{
	deque<shared_ptr<node::Node>> ordered_nodes;

	// DAG topological sorting algorithm with DFS
	for (const auto& node : unordered_nodes)
	{
		order_graph_visit_(node.first, unordered_nodes, ordered_nodes);
	}

	// get input operatos
	for (const auto& node : m_inputs)
	{
		m_executor->add_input_operator(node);
	}

	for (const auto& node : ordered_nodes)
	{
		// node not an input so safe to assume it's an operator
		if (find(
		        m_inputs.begin(), m_inputs.end(),
		        node) == m_inputs.end())
			m_executor->add_operator_node(node);
	}
}

void Graph::order_graph_visit_(
    const shared_ptr<node::Node>& node,
    unordered_map<shared_ptr<node::Node>, Graph::STATUS>& all_nodes,
    deque<shared_ptr<node::Node>>& result)
{
	auto& node_status = all_nodes[node];
	if (node_status == STATUS::MARKED)
		return;
	if (node_status == STATUS::TEMPORARY)
		error("Not a DAG!");

	node_status = STATUS::TEMPORARY;

	for (auto& child_node : node->get_input_nodes())
	{
		order_graph_visit_(child_node, all_nodes, result);
	}

	node_status = STATUS::MARKED;
	result.push_back(node);
}
