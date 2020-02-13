#include <shogun/mathematics/graph/Graph.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/ops/Input.h>

#include <deque>
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
	check_fully_connected(m_inputs, m_outputs);
	build_backend_graph(m_inputs, m_outputs);
}

void Graph::check_fully_connected(
    const std::vector<std::shared_ptr<Input>>& inputs,
    const std::vector<std::shared_ptr<Node>>& outputs)
{
	std::deque<std::shared_ptr<Node>> nodes_to_check;
	for (const auto& node : outputs)
	{
		nodes_to_check.push_back(node);
	}
	std::unordered_set<std::shared_ptr<Node>> inputs_found;

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
	}

	if (inputs_found.size() != inputs.size())
	{
		error("Graph found more input tensors than provided.");
	}
}

void Graph::evaluate(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
	auto* env = ShogunEnv::instance();
	switch (env->graph_backend())
	{
	case GRAPH::SHOGUN:
	{
		execute_shogun();
	}
	break;
	case GRAPH::NGRAPH:
	{
#ifdef USE_NGRAPH
		execute_ngraph();
#else
		error("NGraph execution is not available.");
#endif
	}
	}
}


void Graph::build_backend_graph(
    const std::vector<std::shared_ptr<Input>>& inputs,
    const std::vector<std::shared_ptr<Node>>& outputs)
{
	auto* env = ShogunEnv::instance();
	switch (env->graph_backend())
	{
	case GRAPH::SHOGUN:
	{
		build_shogun_graph();
	}
	break;
	case GRAPH::NGRAPH:
	{
#ifdef USE_NGRAPH
		build_ngraph_graph();
#else
		error("NGraph execution is not available.");
#endif
	}
	}	
}

void Graph::build_shogun_graph() {}
void Graph::build_ngraph_graph() {}


void Graph::get_backend_operator(const std::shared_ptr<Node>& node)
{
	auto* env = ShogunEnv::instance();

	switch (env->graph_backend())
	{
	case GRAPH::NGRAPH:
	{
#ifdef USE_NGRAPH
		// node->s
#endif
	}
	break;
	case GRAPH::XLA:
	case GRAPH::TVM:
	case GRAPH::SHOGUN:
		break;
	}
}


void Graph::execute_shogun()
{
}

void Graph::execute_ngraph()
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
