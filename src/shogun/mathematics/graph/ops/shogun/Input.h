#include <shogun/mathematics/graph/ops/Input.h>
#include <shogun/mathematics/graph/OperatorImplementation.h>

namespace shogun {
	template <typename DerivedOperator, typename EngineImplementation>
    IGNORE_IN_CLASSLIST class InputImpl: public OperatorImpl<EngineImplementation>
	{
	public:
		InputImpl(): OperatorImpl<EngineImplementation>() {}
		// InputImpl(const std::shared_ptr<Input>& node): OperatorImpl<EngineImplementation>(node) {}

		virtual ~InputImpl() {}

		std::string_view get_operator_name() const override
		{
			return "Input";
		}

		void evaluate() override
		{
			error("Input nodes cannot be run with evaluate. Use evaluate_input(SGContainer) instead");
		}

		template<typename T>
		void evaluate_input(const SGVector<T>& vec)
		{
			this->evaluate_implementation(vec);
		}

		template<typename T>
		void evaluate_input(const SGMatrix<T>& vec)
		{
			this->evaluate_implementation(vec);
		}
	};

    IGNORE_IN_CLASSLIST class InputShogun: public InputImpl<InputShogun, OperatorShogunBackend>
	{
	public:
		InputShogun(): InputImpl() {}

		void evaluate(const std::shared_ptr<Tensor>& tensor)
		{		
			auto input_node = std::static_pointer_cast<Input>(m_abstract_node);

			runtime_type_check(tensor->get_type());
			runtime_shape_check(tensor->get_shape());

			// allocate_output(tensor->get_shape(), tensor->get_type());

			input_node->get_tensor()->data() = tensor->data();
		}
		
	private:
		void allocate_output(const Shape& shape, element_type type) {
			auto input_node = std::static_pointer_cast<Input>(m_abstract_node);

			input_node->get_tensor() = std::make_shared<Tensor>(shape, type);
		}

		void runtime_type_check(element_type type)
		{
			// we trust the implementation to only use this implementation
			// when the abstract node is an input
			auto input_node = std::static_pointer_cast<Input>(m_abstract_node);

			const auto& input_tensor = input_node->get_tensor();
			if (type != input_tensor->get_type())
				error("Input node got wrong input type!");
		}

		void runtime_shape_check(const Shape& shape)
		{
			auto input_node = std::static_pointer_cast<Input>(m_abstract_node);

			const auto& input_tensor = input_node->get_tensor();
			const auto expected_shape = input_tensor->get_shape();

			if (shape.size() != expected_shape.size())
			{
				error("Mismatch in the number of dimensions, expected {}, but got {}", 
					expected_shape.size(), shape.size());
			}

			for (const auto& [idx, input_shape_i, expected_shape_i]: enumerate(shape, expected_shape))
			{
				if (expected_shape_i == Shape::Dynamic)
					continue;
				else if (expected_shape_i != input_shape_i)
				{
					error("Runtime shape mismatch in dimension {}. Got {} but expected {}.",
						idx, shape, expected_shape
						);
				}
			}
		}
	};
}