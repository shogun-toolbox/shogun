#include <shogun/mathematics/graph/ShogunGraph.h>
#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/mathematics/graph/runtime/shogun/Add.h>
#include <shogun/mathematics/graph/runtime/shogun/Cast.h>
#include <shogun/mathematics/graph/runtime/shogun/Divide.h>
#include <shogun/mathematics/graph/runtime/shogun/Dot.h>
#include <shogun/mathematics/graph/runtime/shogun/Equal.h>
#include <shogun/mathematics/graph/runtime/shogun/Input.h>
#include <shogun/mathematics/graph/runtime/shogun/LogicalAnd.h>
#include <shogun/mathematics/graph/runtime/shogun/LogicalOr.h>
#include <shogun/mathematics/graph/runtime/shogun/LogicalXor.h>
#include <shogun/mathematics/graph/runtime/shogun/MatMul.h>
#include <shogun/mathematics/graph/runtime/shogun/Multiply.h>
#include <shogun/mathematics/graph/runtime/shogun/Reshape.h>
#include <shogun/mathematics/graph/runtime/shogun/Subtract.h>

using namespace shogun::graph;

OpMapFactory& OperatorRegistry()
{
	static OpMapFactory operator_registry;
	return operator_registry;
}

std::vector<std::shared_ptr<Tensor>> ShogunGraph::execute(
    const std::vector<std::shared_ptr<Tensor>>& tensors,
    const std::vector<std::shared_ptr<node::Node>>& output_nodes) const
{
	if (m_operator_output_nodes.empty())
	{
		error("Did you call Graph::build()?");
	}

	if (m_input_output_nodes.empty())
	{
		error("No input nodes found in graph!");
	}

	if (tensors.size() != m_input_output_nodes.size())
		error(
		    "Number of input tensors ({}) different from number of input nodes "
		    "({}).",
		    tensors.size(), m_input_output_nodes.size());

	for (auto [tensor, node] : zip_iterator(tensors, m_input_output_nodes))
	{
		node->evaluate_tensor(tensor);
	}

	std::for_each(
	    m_operator_output_nodes.begin(), m_operator_output_nodes.end(),
	    [](auto& op) { op->operator()(); });

	std::vector<std::shared_ptr<Tensor>> results;
	for (const auto& node : output_nodes)
	{
		results.push_back(extract_result(node));
	}

	return results;
}

std::shared_ptr<detail::RuntimeNode>
ShogunGraph::get_operator(const std::shared_ptr<node::Node>& node) const
{
	auto type = std::type_index(typeid(*node));
	auto op_it = OperatorRegistry().find(type);
	if (op_it == OperatorRegistry().end())
	{
		error("Could not find operator for node {}", node->to_string());
	}
	return op_it->second();
}

std::shared_ptr<Tensor>
ShogunGraph::extract_result(const std::shared_ptr<node::Node>& node) const
{
	const auto& result = m_lookup.at(node)->get_output_tensors();
	if (result.size() > 1)
		error("Only one tensor for each node can be currently handled");
	if (result.empty())
		error("No output tensor found. Did you call Graph::build()?");

	return result[0];
}

void ShogunGraph::add_input_operator(const std::shared_ptr<node::Node>& node)
{
	auto input = get_operator(node);
	m_lookup[node] =
	    std::static_pointer_cast<detail::shogun::InputShogun>(input)
	        ->build_input(node);
	m_input_output_nodes.push_back(m_lookup.at(node));
}

void ShogunGraph::add_operator_node(const std::shared_ptr<node::Node>& node)
{
	auto op = get_operator(node);
	std::vector<std::shared_ptr<detail::shogun::OutputNode>> inputs;
	for (const auto& input : node->get_input_nodes())
		inputs.push_back(m_lookup.at(input));

	m_lookup[node] = std::static_pointer_cast<detail::RuntimeNodeTemplate<
	    node::Node, detail::shogun::OutputNode>>(op)
	                     ->build(inputs, node);
	m_operator_output_nodes.push_back(m_lookup.at(node));
}

// move this to implementations....
REGISTER_OP(detail::shogun::AddShogun);
REGISTER_OP(detail::shogun::EqualShogun);
REGISTER_OP(detail::shogun::InputShogun);
REGISTER_OP(detail::shogun::SubtractShogun);
REGISTER_OP(detail::shogun::MultiplyShogun);
REGISTER_OP(detail::shogun::DivideShogun);
REGISTER_OP(detail::shogun::LogicalAndShogun);
REGISTER_OP(detail::shogun::LogicalOrShogun);
REGISTER_OP(detail::shogun::LogicalXorShogun);
REGISTER_OP(detail::shogun::DotShogun);
REGISTER_OP(detail::shogun::ReshapeShogun);
REGISTER_OP(detail::shogun::MatMulShogun);
REGISTER_OP(detail::shogun::CastShogun);

BEGIN_EXECUTOR_MANIFEST("Shogun default graph executor")
EXPORT_EXECUTOR(ShogunGraph)
END_EXECUTOR_MANIFEST()
