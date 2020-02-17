/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNBINARYOPERATOR_H_
#define SHOGUNBINARYOPERATOR_H_

#include <shogun/mathematics/graph/ops/Operator.h>
#include <iostream>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename DerivedOperator>
			class ShogunBinaryOperator : public Operator
			{
			public:
				ShogunBinaryOperator(const std::shared_ptr<node::Node>& node): Operator(node)
				{
				}

				virtual ~ShogunBinaryOperator()
				{
				}

				void call(const std::vector<std::shared_ptr<detail::shogun::OutputNode>>& input_nodes) final
				{
					// if (input_nodes.size() != 2)
					// 	error("Expected 2 input nodes in binary operation");

					const auto& input_tensor1 = input_nodes[0]->get_output_tensors()[0];
					const auto& input_tensor2 = input_nodes[1]->get_output_tensors()[0];
					const auto& output_tensor = m_output_tensors[0];

					runtime_checks_and_allocation(std::vector{input_tensor1, input_tensor2});

					kernel(
					    input_tensor1->data(), input_tensor2->data(),
					    output_tensor->data(), output_tensor->size(),
					    output_tensor->get_type());
				}

			protected:
				void runtime_checks_and_allocation(const std::vector<std::shared_ptr<Tensor>>& tensors) final
				{
					if (tensors.size() != 2)
						error("Binary operation expected two inputs.");
					if (m_output_tensors.size() != 1)
						error("Binary operation expected one output.");
					
					const auto& input_tensor1 = tensors[0];
					const auto& input_tensor2 = tensors[1];
					
					allocate_tensor(runtime_shape_check(input_tensor1, input_tensor2));
				}

				void allocate_tensor(const Shape& shape)
				{
					m_output_tensors[0]->allocate_tensor(shape);
				}

				const Shape& runtime_shape_check(
				    const std::shared_ptr<Tensor>& tensor1,
				    const std::shared_ptr<Tensor>& tensor2)
				{
					for (auto [idx, shape1, shape2] : enumerate(
					         tensor1->get_shape(), tensor2->get_shape()))
					{
						if (shape1 != shape2)
						{
							error(
							    "Runtime shape mismatch in dimension {}. Got {} and "
							    "{}.",
							    idx, shape1, shape2);
						}
						if (shape1 == Shape::Dynamic)
						{
							error("Could not infer runtime shape.");
						}
					}

					// shapes have to match exeactly so can return either one
					return tensor1->get_shape();
				}

				void kernel(
				    void* input1, void* input2, void* output, size_t size,
				    element_type type)
				{
					switch (type)
					{
					case element_type::FLOAT32:
						static_cast<DerivedOperator*>(this)
						    ->template kernel_implementation<
						        get_type_from_enum<element_type::FLOAT32>::type>(
						        input1, input2, output, size);
						break;
					case element_type::FLOAT64:
						static_cast<DerivedOperator*>(this)
						    ->template kernel_implementation<
						        get_type_from_enum<element_type::FLOAT64>::type>(
						        input1, input2, output, size);
						break;
					}
				}
			};
		}
	}
} // namespace shogun

#endif