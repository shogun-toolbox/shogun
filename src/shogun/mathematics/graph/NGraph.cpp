#include <shogun/mathematics/graph/NGraph.h>
#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/ops/ngraph/Add.h>

#include <ngraph/ngraph.hpp>

using namespace shogun;

OpMapFactory& OperatorRegistry()
{
	static OpMapFactory operator_registry;
	return operator_registry;
}

void NGraph::execute(const std::vector<std::shared_ptr<Tensor>>& tensors) const
{
	auto backend = ngraph::runtime::Backend::create("CPU", true);

	/*
	std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	    ngraph_input_tensors;
	std::vector<std::shared_ptr<ngraph::runtime::Tensor>>
	    ngraph_output_tensors;

	for (const auto& tensor : input_tensors)
		ngraph_input_tensors.push_back(backend->create_tensor(
		    ngraph::element::f32, tensor.get_shape()));
	for (const auto& tensor : output_tensors)
		ngraph_output_tensors.push_back(backend->create_tensor(
		    ngraph::element::f32, tensor.get_shape()));

	auto handle = backend->compile(graph->get_ngraph_function());
	handle->call_with_validate(
	    ngraph_input_tensors, ngraph_output_tensors);

	std::vector<Tensor> results;
	for (const auto& ngraph_tensor : ngraph_output_tensors)
	{
		results.push_back(Tensor::create_empty(
		    ngraph_tensor->get_shape(),
		    get_enum_from_ngraph(ngraph_tensor->get_element_type())));
		ngraph_tensor->read(
		    results.back().data(),
		    results.back().get_size() * sizeof(float));
	}

	return results
	*/
}

void NGraph::add_input_operator(const std::shared_ptr<Node>& node)
{
}

void NGraph::add_operator_node(const std::shared_ptr<Node>& node)
{
}

REGISTER_OP(AddNGraph);

BEGIN_EXECUTOR_MANIFEST("NGraph based graph executor")
EXPORT_EXECUTOR(NGraph)
END_EXECUTOR_MANIFEST()
