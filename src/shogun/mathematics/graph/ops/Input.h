/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUT_H_
#define SHOGUNINPUT_H_

#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/OperationImplementation.h>

#include <shogun/util/enumerate.h>

#define IGNORE_IN_CLASSLIST

namespace shogun {

	IGNORE_IN_CLASSLIST class Input: public Node
	{
	public:
		Input(const Shape& shape, element_type type): Node(shape, type) {
		}

		std::shared_ptr<Tensor>& get_tensor()
		{
			// we know that the is only one output here
			return m_output_tensors[0];
		}

		const std::shared_ptr<Tensor>& get_tensor() const
		{
			// we know that the is only one output here
			return m_output_tensors[0];
		}

		void allocate_tensor(const shogun::Shape& shape, shogun::element_type type) {
			m_output_tensors[0]->data() = allocator_dispatch(m_output_tensors[0]->get_size_from_shape(shape), type);
		}

		std::string to_string() const override
		{
			return fmt::format("Input(shape={}, type={})", get_tensors()[0]->get_shape(), get_tensors()[0]->get_type());
		}

	};

	template <typename EngineImplementation>
    IGNORE_IN_CLASSLIST class InputImpl: public OperationImpl
	{
	public:
		InputImpl(const std::shared_ptr<Input>& node): OperationImpl(node) {}

		virtual ~InputImpl() {}

		template<typename T>
		void evaluate(const SGVector<T>& vec)
		{
			this->evaluate_implementation(vec);
		}

		template<typename T>
		void evaluate(const SGMatrix<T>& vec)
		{
			this->evaluate_implementation(vec);
		}
	};

    IGNORE_IN_CLASSLIST class ShogunInput: public InputImpl<ShogunInput>
	{
	public:
		ShogunInput(const std::shared_ptr<Input>& node): InputImpl(node) {}

		void build()
		{
		}

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

#endif