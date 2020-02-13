/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef LINALGNODES_H_
#define LINALGNODES_H_

#include <shogun/mathematics/graph/Shape.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/Types.h>

#ifdef USE_NGRAPH
#include <ngraph/ngraph.hpp>
#endif

namespace shogun
{
	// The node classes
	class Node
	{
	public:
		Node(const Shape& shape, element_type type): m_tensor(std::make_shared<Tensor>(shape, type)) {}

		Node(const std::shared_ptr<Tensor>& tensor): m_tensor(tensor) {}

		virtual void evaluate() = 0;

		virtual ~Node()
		{
		}

#ifdef USE_NGRAPH
		void set_ngraph(std::shared_ptr<ngraph::Node> node)
		{
			m_ngraph_node = std::move(node);
		}

		const std::shared_ptr<ngraph::Node>& get_ngraph() const
		{
			return m_ngraph_node;
		}
#endif
		const std::shared_ptr<Tensor>& get_tensor() const
		{
			return m_tensor;
		}

	protected:
		std::shared_ptr<Tensor> m_tensor;

	private:
		virtual void allocate_tensor(const Shape& shape, element_type type) = 0;
#ifdef USE_NGRAPH
		std::shared_ptr<ngraph::Node> m_ngraph_node;
#endif
	};

	// TODO
	class Graph
	{
	public:
		Graph();
		~Graph();
		std::vector<Tensor> execute_shogun(
		    const std::vector<Tensor>& input_tensors,
		    const std::vector<Tensor>& output_tensors);
#ifdef USE_NGRAPH
		std::shared_ptr<ngraph::Function> get_ngraph_function();
#endif
	};

	// most of this could live in the graph class
	std::vector<Tensor> evaluate(
	    const std::vector<Tensor>& input_tensors,
	    const std::vector<Tensor>& output_tensors,
	    const std::shared_ptr<Graph>& graph)
	{
		auto* env = ShogunEnv::instance();
		switch (env->graph_backend())
		{
		case GRAPH::SHOGUN:
		{
			return graph->execute_shogun(input_tensors, output_tensors);
		}
		break;
#ifdef USE_NGRAPH
		case GRAPH::NGRAPH:
		{
			auto backend = ngraph::runtime::Backend::create("CPU", true);

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

			return results;
		}
#endif
		}
	}
} // namespace shogun

#endif