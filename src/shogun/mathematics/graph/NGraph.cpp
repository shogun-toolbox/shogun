#include <shogun/mathematics/graph/NGraph.h>
#include <shogun/mathematics/graph/node_implementation/ngraph/Add.h>
#include <shogun/mathematics/graph/node_implementation/ngraph/Input.h>
#include <shogun/mathematics/graph/node_implementation/ngraph/Subtract.h>
#include <shogun/mathematics/graph/node_implementation/ngraph/Multiply.h>
#include <shogun/mathematics/graph/node_implementation/ngraph/Divide.h>
#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/mathematics/graph/Types.h>
#include <shogun/mathematics/graph/Shape.h>

#include <ngraph/ngraph.hpp>

using namespace shogun::graph;
using namespace shogun::graph::detail::ngraph;

OpMapFactory& OperatorRegistry()
{
	static OpMapFactory operator_registry;
	return operator_registry;
}

std::vector<std::shared_ptr<Tensor>> NGraph::execute(const std::vector<std::shared_ptr<Tensor>>& tensors, 
	const std::vector<std::shared_ptr<node::Node>>& output_nodes) const
{
	auto backend = ngraph::runtime::Backend::create("CPU", true);
	
	std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	    ngraph_input_tensors;
	std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	    ngraph_output_tensors;

	for (const auto& tensor: tensors)
	{
		const auto& shape = tensor->get_shape();
	    ngraph_input_tensors.push_back(backend->create_tensor(
	        get_ngraph_type_from_enum(tensor->get_type()), 
	        to_ngraph_shape(shape)));
		ngraph_input_tensors.back()->write(tensor->data(), tensor->size_in_bytes());

	}
	
	ngraph::ParameterVector inputs;
	ngraph::OutputVector outputs;

	for (const auto& el: m_input_output_nodes)
	{
		inputs.push_back(std::static_pointer_cast<ngraph::op::Parameter>(el));
	}

	for (const auto& el: output_nodes)
	{
		outputs.push_back(m_lookup.at(el));
	}

	auto f = std::make_shared<ngraph::Function>(outputs, inputs);

	auto handle = backend->compile(f);

	for (const auto& node : output_nodes)
	{
		const auto& shape = node->get_shapes()[0];
		const auto& type = node->get_types()[0];

		if (std::find(shape.begin(), shape.end(), Shape::Dynamic) != shape.end())
		{
		    ngraph_output_tensors.push_back(backend->create_dynamic_tensor(
		        get_ngraph_type_from_enum(type), 
		        to_ngraph_partial_shape(shape)));	
		}
		else
		{
		    ngraph_output_tensors.push_back(backend->create_tensor(
		        get_ngraph_type_from_enum(type), 
		        to_ngraph_shape(shape)));
		}
	}

	handle->call_with_validate(
	    ngraph_output_tensors, ngraph_input_tensors);

	std::vector<std::shared_ptr<Tensor>> results;
	for (const auto& ngraph_tensor : ngraph_output_tensors)
	{
		const auto shape = from_ngraph_shape(ngraph_tensor->get_shape());
		const auto type = get_enum_from_ngraph(ngraph_tensor->get_element_type());

	    auto& tensor = results.emplace_back(std::make_shared<Tensor>(shape, type));
	    tensor->allocate_tensor(shape);
	    ngraph_tensor->wait_for_read_ready();
	    ngraph_tensor->read(
	        tensor->data(),
	        ngraph_tensor->get_size_in_bytes());
	}

	return results;
}

std::shared_ptr<detail::RuntimeNode>
NGraph::get_operator(const std::shared_ptr<node::Node>& node) const
{
	auto type = std::type_index(typeid(*node));
	auto op_it = OperatorRegistry().find(type);
	// std::cout << type.name() << '\n';
	// for(const auto& el: OperatorRegistry())
	// 	std::cout << el.first.name() << '\n';
	if (op_it == OperatorRegistry().end())
	{
		error("Could not find operator for node {}", node->to_string());
	}
	return op_it->second();
}

void NGraph::add_input_operator(const std::shared_ptr<node::Node>& node)
{

	auto input = get_operator(node);
	m_lookup[node] = std::static_pointer_cast<detail::ngraph::InputNGraph>(input)->build_input(node);
	m_input_output_nodes.push_back(m_lookup.at(node));
}

void NGraph::add_operator_node(const std::shared_ptr<node::Node>& node)
{
	auto op = get_operator(node);
	std::vector<std::shared_ptr<ngraph::Node>> inputs;
	for (const auto& input: node->get_input_nodes())
		inputs.push_back(m_lookup.at(input));

	m_lookup[node] =
	    std::static_pointer_cast<
	    detail::RuntimeNodeTemplate<node::Node, ngraph::Node>>(op)->build(inputs, node);
	m_operator_output_nodes.push_back(m_lookup.at(node));
}

REGISTER_OP(detail::ngraph::AddNGraph);
REGISTER_OP(detail::ngraph::DivideNGraph);
REGISTER_OP(detail::ngraph::InputNGraph);
REGISTER_OP(detail::ngraph::MultiplyNGraph);
REGISTER_OP(detail::ngraph::SubtractNGraph);


BEGIN_EXECUTOR_MANIFEST("NGraph based graph executor")
EXPORT_EXECUTOR(NGraph)
END_EXECUTOR_MANIFEST()
