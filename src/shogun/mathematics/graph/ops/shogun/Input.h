/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUTSHOGUN_H_
#define SHOGUNINPUTSHOGUN_H_

#include <shogun/mathematics/graph/ops/abstract/InputImpl.h>
#include <shogun/mathematics/graph/nodes/Input.h>

namespace shogun {

    IGNORE_IN_CLASSLIST class InputShogun: public InputImpl<InputShogun>
	{
	public:
		InputShogun(): InputImpl() {}

		void evaluate_implementation(const std::shared_ptr<Tensor>& tensor)
		{
			auto input_node = std::static_pointer_cast<operator_type>(m_abstract_node);

			runtime_type_check(tensor->get_type());
			runtime_shape_check(tensor->get_shape());

			allocate_output(tensor->get_shape(), tensor->get_type());

			// the input node is just a handle to the input tensor
			input_node->get_tensor()->data() = tensor->data();
		}

		void build_implementation() final
		{
		}

	private:
		void allocate_output(const Shape& shape, element_type type) {
			auto input_node = std::static_pointer_cast<operator_type>(m_abstract_node);
			input_node->get_tensor() = std::make_shared<Tensor>(shape, type);
		}

		void runtime_type_check(element_type type)
		{
			// we trust the implementation to only use this implementation
			// when the abstract node is an input
			auto input_node = std::static_pointer_cast<operator_type>(m_abstract_node);

			const auto& input_tensor = input_node->get_tensor();
			if (type != input_tensor->get_type())
				error("Input node got wrong input type!");
		}

		void runtime_shape_check(Shape shape)
		{
			auto input_node = std::static_pointer_cast<operator_type>(m_abstract_node);

			const auto& input_tensor = input_node->get_tensor();
			const auto expected_shape = input_tensor->get_shape();

			if (shape.size() != expected_shape.size())
			{
				error("Mismatch in the number of dimensions, expected {}, but got {}",
					expected_shape.size(), shape.size());
			}

			for (const auto& [idx, input_shape_i, expected_shape_i]: enumerate(shape, expected_shape))
			{
				// if it is dynamic we will use this to infer the name shape
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