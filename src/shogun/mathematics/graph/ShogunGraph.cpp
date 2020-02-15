#include <shogun/mathematics/graph/ShogunGraph.h>
#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/ops/shogun/Add.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>
#include <shogun/mathematics/graph/ops/shogun/Subtract.h>

using namespace shogun;

OpMapFactory& OperatorRegistry()
{
	static OpMapFactory operator_registry;
	return operator_registry;
}

void ShogunGraph::execute(const std::vector<std::shared_ptr<Tensor>>& tensors) const
{
    if (m_cached_operators.empty())
	{
		error("Did you call Graph::build()?");
	}

	if (m_cached_input_operators.empty())
	{
		error("No input nodes found in graph!");
	}

	if (tensors.size() != m_cached_input_operators.size())
		error("Number of input tensors ({}) different from number of input nodes ({}).",
			tensors.size(), m_cached_input_operators.size());
	for (const auto& [tensor, node]: zip_iterator(tensors, m_cached_input_operators))
	{
		std::static_pointer_cast<InputShogun>(node)->evaluate_input(tensor);
	}

	std::for_each(m_cached_operators.begin(), m_cached_operators.end(),
		[] (auto& op) { (*op)(); });

}

std::shared_ptr<Operator> ShogunGraph::get_operator(const std::shared_ptr<Node>& node) const
{
	auto type = std::type_index(typeid(*node));
	auto op_it = OperatorRegistry().find(type);
	if (op_it == OperatorRegistry().end())
	{
		error("Unsupported OP type");
	}
	return op_it->second();
}

void ShogunGraph::add_input_operator(const std::shared_ptr<Node>& node)
{
	auto input = get_operator(node);
    input->build(node);
	m_cached_input_operators.push_back(input);
}

void ShogunGraph::add_operator_node(const std::shared_ptr<Node>& node)
{
	auto op = get_operator(node);
    op->build(node);
    m_cached_operators.push_back(op);
}

// move this to implementations....
REGISTER_OP(AddShogun);
REGISTER_OP(InputShogun);
REGISTER_OP(SubtractShogun);


BEGIN_EXECUTOR_MANIFEST("Shogun default graph executor")
EXPORT_EXECUTOR(ShogunGraph)
END_EXECUTOR_MANIFEST()
