/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNINPUT_H_
#define SHOGUNINPUT_H_

#include <shogun/mathematics/graph/LinalgNodes.h>
#include <shogun/mathematics/graph/Tensor.h>
#include <shogun/mathematics/graph/OperatorImplementation.h>

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

		std::string_view get_operator_name() const override
		{
			return "Input";
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
}

#endif