/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef LINALGNODES_H_
#define LINALGNODES_H_

#include <shogun/lib/SGVector.h>
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
		Node(const std::vector<size_t>& shape, element_type type)
		    : m_shape(shape), m_type(type)
		{
		}

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
		void* get_data() const
		{
			return m_data;
		}

		const std::vector<size_t>& get_shape() const
		{
			return m_shape;
		}

		element_type get_type() const
		{
			return m_type;
		}

		size_t get_size_from_shape(const std::vector<size_t>& size)
		{
			return std::accumulate(
			    size.begin(), size.end(), 1, std::multiplies{});
		}

	protected:
		void* m_data;
		const std::vector<size_t> m_shape;
		const element_type m_type;

	private:
		virtual void allocate_data() = 0;
#ifdef USE_NGRAPH
		std::shared_ptr<ngraph::Node> m_ngraph_node;
#endif
	};

	// simple add node, could be have option for inplace?
	class Add : public Node
	{
	public:
		Add(const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		    : m_node1(node1), m_node2(node2),
		      Node(
		          check_shape_compatible(node1, node2),
		          check_type_compatible(node1, node2))
		{
			auto* env = ShogunEnv::instance();

			switch (env->graph_backend())
			{
			case GRAPH::NGRAPH:
			{
#ifdef USE_NGRAPH
				set_ngraph(std::make_shared<ngraph::op::Add>(
				    node1->get_ngraph(), node2->get_ngraph()));
#endif
			}
			break;
			case GRAPH::XLA:
			case GRAPH::TVM:
			case GRAPH::SHOGUN:
				break;
			}
		}

		void evaluate()
		{
			auto* env = ShogunEnv::instance();

			// node evaluation happens in each engine's implementation
			// here we just need the shogun version
			switch (env->graph_backend())
			{
			case GRAPH::SHOGUN:
			{
				add(m_node1, m_node2);
			}
			break;
			case GRAPH::NGRAPH:
			case GRAPH::XLA:
			case GRAPH::TVM:
				break;
			}
		}

	protected:
		std::shared_ptr<Node> m_node1;
		std::shared_ptr<Node> m_node2;

	private:
		void
		add(const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			allocate_data();
			// the actual call
			add_kernel(
			    node1->get_data(), node2->get_data(), m_data,
			    get_size_from_shape(get_shape()), m_type);
		}

		element_type check_type_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			if (m_node1->get_type() != m_node2->get_type())
				error("Expected types to be the same");
			return m_node1->get_type();
		}

		std::vector<size_t> check_shape_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			if (m_node1->get_shape() != m_node2->get_shape())
				error("Incompatible shapes");
			return m_node1->get_shape();
		}

		void allocate_data()
		{
			m_data =
			    allocator_dispatch(get_size_from_shape(get_shape()), m_type);
		}

		template <typename T>
		void
		add_kernel_helper(void* input1, void* input2, void* output, size_t size)
		{
			// if we have SYCL or MSVC we could add parallel execution
			// or just use Eigen here
			std::transform(
			    static_cast<const T*>(input1),
			    static_cast<const T*>(input1) + size,
			    static_cast<const T*>(input2), static_cast<T*>(output),
			    std::plus<T>());
		}

		void add_kernel(
		    void* input1, void* input2, void* output, size_t size,
		    element_type type)
		{
			switch (type)
			{
			case element_type::FLOAT32:
				add_kernel_helper<
				    get_type_from_enum<element_type::FLOAT32>::type>(
				    input1, input2, output, size);
				break;
			case element_type::FLOAT64:
				add_kernel_helper<
				    get_type_from_enum<element_type::FLOAT64>::type>(
				    input1, input2, output, size);
				break;
			}
		}
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
		std::shared_ptr<ngraph::Function> get_ngraph_function();
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