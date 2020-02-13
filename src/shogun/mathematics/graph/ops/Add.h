#include <shogun/mathematics/graph/LinalgNodes.h>

namespace shogun {
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
			allocate_tensor(check_shape_compatible(node1, node2), 
				check_type_compatible(node1, node2));
			// the actual call
			kernel(
			    node1->get_tensor()->data(), node2->get_tensor()->data(), m_tensor->data(),
			    m_tensor->size(), m_tensor->get_type());
		}

		element_type check_type_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			if (m_node1->get_tensor()->get_type() != m_node2->get_tensor()->get_type())
				error("Expected types to be the same");
			return m_node1->get_tensor()->get_type();
		}

		Shape check_shape_compatible(
		    const std::shared_ptr<Node>& node1,
		    const std::shared_ptr<Node>& node2)
		{
			if (m_node1->get_tensor()->get_shape().size() != m_node2->get_tensor()->get_shape().size())
			{
				error("Number of dimension mismatch between {} and {}.", m_node1, m_node2);
			}
			// for (const auto& [shape1, shape2]: zip_iterator(m_node1->get_shape(), m_node2->get_shape()))
			// 	if (shape1 != shape2)


			return m_node1->get_tensor()->get_shape();
		}

		void allocate_tensor(const Shape& shape, element_type type)
		{
			m_tensor = std::make_shared<Tensor>(shape, type);
		}

		void kernel(
		    void* input1, void* input2, void* output, size_t size,
		    element_type type)
		{
			switch (type)
			{
			case element_type::FLOAT32:
				kernel_helper<
				    get_type_from_enum<element_type::FLOAT32>::type>(
				    input1, input2, output, size);
				break;
			case element_type::FLOAT64:
				kernel_helper<
				    get_type_from_enum<element_type::FLOAT64>::type>(
				    input1, input2, output, size);
				break;
			}
		}

		template <typename T>
		void
		kernel_helper(void* input1, void* input2, void* output, size_t size)
		{
			// if we have SYCL or MSVC we could add parallel execution
			// or just use Eigen here
			std::transform(
			    static_cast<const T*>(input1),
			    static_cast<const T*>(input1) + size,
			    static_cast<const T*>(input2), static_cast<T*>(output),
			    std::plus<T>());
		}
	};
}