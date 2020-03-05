#include <shogun/mathematics/graph/NGraph.h>
#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/nodes/Node.h>
#include <shogun/mathematics/graph/runtime/ngraph/Input.h>

#include <ngraph/ngraph.hpp>

using namespace shogun::graph;
using namespace shogun::graph::detail;
using namespace shogun::graph::detail::ngraph;

std::vector<std::shared_ptr<Tensor>> NGraph::execute(
    const std::vector<std::shared_ptr<Tensor>>& tensors,
    const std::vector<std::shared_ptr<node::Node>>& output_nodes) const
{
	auto backend = ::ngraph::runtime::Backend::create("CPU", true);

	std::vector<std::shared_ptr<::ngraph::runtime::Tensor>>
	    ngraph_input_tensors;
	std::vector<std::shared_ptr<::ngraph::runtime::Tensor>>
	    ngraph_output_tensors;

	for (const auto& tensor : tensors)
	{
		const auto& shape = tensor->get_shape();
		if (tensor->get_shape().size() < 2 || !m_requires_major_conversion)
		{
			auto& input =
			    ngraph_input_tensors.emplace_back(backend->create_tensor(
			        get_ngraph_type_from_enum(tensor->get_type()),
			        to_ngraph_shape(shape)));
			input->write(tensor->data(), tensor->size_in_bytes());
		}
		else if (tensor->get_shape().size() == 2 && m_requires_major_conversion)
		{
			const auto ngraph_shape = to_ngraph_shape(shape);
			auto& input =
			    ngraph_input_tensors.emplace_back(backend->create_tensor(
			        get_ngraph_type_from_enum(tensor->get_type()),
			        ::ngraph::Shape{ngraph_shape[1], ngraph_shape[0]}));
			input->write(tensor->data(), tensor->size_in_bytes());
			auto& transpose_param =
			    ngraph_input_tensors.emplace_back(backend->create_tensor(
			        ::ngraph::element::i64, ::ngraph::Shape{2}));
			transpose_param->write(&kTranspose, 2 * sizeof(int64_t));
		}
		else
		{
			error("NGraph interface cannot handle tensors with more than 2 "
			      "dimensions.");
		}
	}

	::ngraph::ParameterVector inputs;
	::ngraph::OutputVector outputs;

	for (const auto& node : m_input_output_nodes)
	{
		auto input_node =
		    std::static_pointer_cast<::ngraph::op::Parameter>(node);
		inputs.push_back(input_node);
	}

	for (const auto& el : output_nodes)
	{
		const auto& shape = el->get_shapes()[0];
		const auto& ngraph_shape = to_ngraph_shape(shape);

		if (shape.size() < 2 || !m_requires_major_conversion)
		{
			outputs.push_back(m_lookup.at(el));
		}
		else if (shape.size() == 2 && m_requires_major_conversion)
		{
			auto perm = std::make_shared<::ngraph::op::Parameter>(
			    ::ngraph::element::i64, ::ngraph::Shape{2});
			outputs.push_back(std::make_shared<::ngraph::op::Transpose>(
			    m_lookup.at(el), perm));
			inputs.push_back(perm);
			auto& transpose_param =
			    ngraph_input_tensors.emplace_back(backend->create_tensor(
			        ::ngraph::element::i64, ::ngraph::Shape{2}));
			transpose_param->write(&kTranspose, 2 * sizeof(int64_t));
		}
		else
		{
			error("NGraph interface cannot handle tensors with more than 2 "
			      "dimensions.");
		}
	}

	auto f = std::make_shared<::ngraph::Function>(outputs, inputs);

	auto handle = backend->compile(f);

	for (const auto& node : output_nodes)
	{
		const auto shape = m_requires_major_conversion
		                       ? node->get_shapes()[0].switch_major()
		                       : node->get_shapes()[0];
		const auto& type = node->get_types()[0];

		if (std::find(shape.begin(), shape.end(), Shape::Dynamic) !=
		    shape.end())
		{
			ngraph_output_tensors.push_back(backend->create_dynamic_tensor(
			    get_ngraph_type_from_enum(type),
			    to_ngraph_partial_shape(shape)));
		}
		else
		{
			ngraph_output_tensors.push_back(backend->create_tensor(
			    get_ngraph_type_from_enum(type), to_ngraph_shape(shape)));
		}
	}

	handle->call(ngraph_output_tensors, ngraph_input_tensors);

	std::vector<std::shared_ptr<Tensor>> results;
	for (const auto& ngraph_tensor : ngraph_output_tensors)
	{
		const auto shape = from_ngraph_shape(ngraph_tensor->get_shape());
		const auto type =
		    get_enum_from_ngraph(ngraph_tensor->get_element_type());

		auto& tensor =
		    results.emplace_back(std::make_shared<Tensor>(shape, type));
		tensor->allocate_tensor(shape);
		ngraph_tensor->wait_for_read_ready();
		ngraph_tensor->read(tensor->data(), ngraph_tensor->get_size_in_bytes());
	}

	return results;
}

std::shared_ptr<detail::RuntimeNode>
NGraph::get_operator(const std::shared_ptr<node::Node>& node) const
{
	auto type = std::type_index(typeid(*node));
	auto op_it = NGraphOperatorRegistry().find(type);
	if (op_it == NGraphOperatorRegistry().end())
	{
		error("Could not find operator for node {}", node->to_string());
	}
	return op_it->second();
}

void NGraph::add_input_operator(const std::shared_ptr<node::Node>& node)
{
	auto input = get_operator(node);
	const auto shape = node->get_shapes()[0];

	if (shape.size() < 2 || !m_requires_major_conversion)
	{
		m_lookup[node] =
		    std::static_pointer_cast<detail::ngraph::InputNGraph>(input)
		        ->build_input(node);
		m_input_output_nodes.push_back(m_lookup.at(node));
	}
	else if (shape.size() == 2 && m_requires_major_conversion)
	{
		auto input_node = std::make_shared<::ngraph::op::Parameter>(
		    get_ngraph_type_from_enum(node->get_types()[0]),
		    to_ngraph_partial_shape(Shape{shape[1], shape[0]}));
		auto perm = std::make_shared<::ngraph::op::Parameter>(
		    ::ngraph::element::i64, ::ngraph::Shape{2});
		m_lookup[node] =
		    std::make_shared<::ngraph::op::Transpose>(input_node, perm);
		m_input_output_nodes.push_back(input_node);
		m_input_output_nodes.push_back(perm);
	}
	else
	{
		error("NGraph interface cannot handle tensors with more than 2 "
		      "dimensions.");
	}
}

void NGraph::add_operator_node(const std::shared_ptr<node::Node>& node)
{
	auto op = get_operator(node);
	std::vector<std::shared_ptr<::ngraph::Node>> inputs;
	for (const auto& input : node->get_input_nodes())
		inputs.push_back(m_lookup.at(input));

	m_lookup[node] =
	    std::static_pointer_cast<
	        detail::RuntimeNodeTemplate<node::Node, ::ngraph::Node>>(op)
	        ->build(inputs, node);
	m_operator_output_nodes.push_back(m_lookup.at(node));
}

REGISTER_OP_NGRAPH(detail::ngraph::InputNGraph);

BEGIN_EXECUTOR_MANIFEST("NGraph based graph executor")
EXPORT_EXECUTOR(NGraph)
END_EXECUTOR_MANIFEST()
