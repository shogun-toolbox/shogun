/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNBINARYOPERATOR_H_
#define SHOGUNBINARYOPERATOR_H_

#include <shogun/mathematics/graph/ops/abstract/OperatorImplementation.h>

namespace shogun {

	template <typename DerivedOperator, typename InterfaceOperator>
	class ShogunBinaryOperator : public OperatorImpl<InterfaceOperator>
	{
	public:
		ShogunBinaryOperator(): OperatorImpl<InterfaceOperator>() {}

		virtual ~ShogunBinaryOperator() {}

		void evaluate() override
		{
			const auto& node1 = this->m_abstract_node->get_input_nodes()[0];
			const auto& node2 = this->m_abstract_node->get_input_nodes()[1];

			auto input_tensor1 = node1->get_tensors()[0];
			auto input_tensor2 = node2->get_tensors()[0];
			auto output_tensor = this->m_abstract_node->get_tensors()[0];

			allocate_tensor(runtime_shape_check(node1, node2));

			kernel(
			    input_tensor1->data(),
			    input_tensor2->data(),
			    output_tensor->data(), output_tensor->size(),
			    output_tensor->get_type());
		}
	private:

		void allocate_tensor(const Shape& shape)
		{
			this->m_abstract_node->get_tensors()[0]->allocate_tensor(shape);
		}

		const Shape& runtime_shape_check(const std::shared_ptr<Node>& node1,
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
				if (shape1 == Shape::Dynamic)
				{
					error("Could not infer runtime shape.");
				}
			}

			return node1_tensor->get_shape();
		}

	protected:
		void kernel(
		    void* input1, void* input2, void* output, size_t size,
		    element_type type)
		{
			switch (type)
			{
			case element_type::FLOAT32:
				static_cast<DerivedOperator*>(this)->template kernel_implementation<
				    get_type_from_enum<element_type::FLOAT32>::type>(
				    input1, input2, output, size);
				break;
			case element_type::FLOAT64:
				static_cast<DerivedOperator*>(this)->template kernel_implementation<
				    get_type_from_enum<element_type::FLOAT64>::type>(
				    input1, input2, output, size);
				break;
			}
		}
	};
}

#endif