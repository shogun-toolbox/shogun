#include <shogun/mathematics/graph/ops/Add.h>
#include <shogun/mathematics/graph/OperatorImplementation.h>

namespace shogun {
	
	template <typename DerivedOperator, typename EngineImplementation>
	IGNORE_IN_CLASSLIST class AddImpl: public OperatorImpl<EngineImplementation>
	{
	public:
		AddImpl(): OperatorImpl<EngineImplementation>() {}
		// AddImpl(const std::shared_ptr<Node>& node): OperatorImpl<EngineImplementation>(node) {}

		virtual ~AddImpl() {}
		
		std::string_view get_operator_name() const override
		{
			return "Add";
		}
		
		template<typename T>
		void evaluate()
		{
			this->evaluate_implementation();
		}
	};


	IGNORE_IN_CLASSLIST class AddShogun : public AddImpl<AddShogun, OperatorShogunBackend>
	{
	public:

		AddShogun(): AddImpl() {};

		// AddShogun(const std::shared_ptr<Node>& node) : AddImpl(node)
		// {
		// }

		void build(const std::shared_ptr<Node>& node)
		{
			m_abstract_node = node;
		}

		void evaluate()
		{
			auto add_node = std::static_pointer_cast<Add>(m_abstract_node);

			const auto& node1 = m_abstract_node->get_input_nodes()[0];
			const auto& node2 = m_abstract_node->get_input_nodes()[1];

			add_node->allocate_tensor(
			    rutime_shape_check(node1, node2),
			    node1->get_tensors()[0]->get_type());

			auto input_tensor1 = node1->get_tensors()[0];
			auto input_tensor2 = node2->get_tensors()[0];
			auto output_tensor = add_node->get_tensors()[0];


			// the actual call
			kernel(
			    input_tensor1->data(),
			    input_tensor2->data(),
			    output_tensor->data(), output_tensor->size(),
			    output_tensor->get_type());
		}

	private:

		const Shape& rutime_shape_check(const std::shared_ptr<Node>& node1, 
			const std::shared_ptr<Node>& node2)
		{
			// we don't need to check how many tensors there are
			// the compile time check already did that
			// we also know that the number of dimensions match
			const auto& node1_tensor = node1->get_tensors()[0];
			const auto& node2_tensor = node2->get_tensors()[0];

			for (const auto& [idx, shape1, shape2] : enumerate(
			         node1_tensor->get_shape(),
			         node2_tensor->get_shape()))
			{
				if (shape1 != shape2)
				{
					error("Runtime shape mismatch in dimension {}. Got {} and {}.",
						idx, shape1, shape2);
				}
			}

			return node1_tensor->get_shape();
		}

		void kernel(
		    void* input1, void* input2, void* output, size_t size,
		    element_type type)
		{
			switch (type)
			{
			case element_type::FLOAT32:
				kernel_implementation<
				    get_type_from_enum<element_type::FLOAT32>::type>(
				    input1, input2, output, size);
				break;
			case element_type::FLOAT64:
				kernel_implementation<
				    get_type_from_enum<element_type::FLOAT64>::type>(
				    input1, input2, output, size);
				break;
			}
		}

		template <typename T>
		void kernel_implementation(
		    void* input1, void* input2, void* output, size_t size)
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