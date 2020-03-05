/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_UNARY_OPERATOR_H_
#define SHOGUN_UNARY_OPERATOR_H_

#include <shogun/mathematics/graph/ops/abstract/Operator.h>
#include <shogun/mathematics/graph/runtime/shogun/OutputNode.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename DerivedOperator>
			class ShogunUnaryOperator : public Operator
			{
			public:
				ShogunUnaryOperator(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				virtual ~ShogunUnaryOperator()
				{
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					if (input_nodes.size() != 1)
						error("Unary operation expected one inputs.");
					if (m_output_tensors.size() != 1)
						error("Unary operation expected one output.");

					const auto& input_tensor =
					    input_nodes[0]->get_output_tensors()[0];
					const auto& output_tensor = m_output_tensors[0];

					allocate_tensor(runtime_shape_check(input_tensor));

					kernel(
					    input_tensor->data(), output_tensor->data(),
					    output_tensor->size(), output_tensor->get_type());
				}

			protected:
				void allocate_tensor(const Shape& shape)
				{
					m_output_tensors[0]->allocate_tensor(shape);
				}

				const Shape&
				runtime_shape_check(const std::shared_ptr<Tensor>& tensor)
				{
					for (const auto& shape : tensor->get_shape())
					{
						if (shape == Shape::Dynamic)
						{
							error("Could not infer runtime shape.");
						}
					}
					return tensor->get_shape();
				}

				void kernel(
				    void* input, void* output, size_t size, element_type type)
				{

#define CALL_KERNEL_IMPLEMENTATION(SHOGUN_TYPE)                                \
	static_cast<DerivedOperator*>(this)                                        \
	    ->template kernel_implementation<                                      \
	        get_type_from_enum<SHOGUN_TYPE>::type>(input, output, size);       \
	break;

					switch (type)
					{
					case element_type::BOOLEAN:
						CALL_KERNEL_IMPLEMENTATION(element_type::BOOLEAN)
					case element_type::INT8:
						CALL_KERNEL_IMPLEMENTATION(element_type::INT8)
					case element_type::INT16:
						CALL_KERNEL_IMPLEMENTATION(element_type::INT16)
					case element_type::INT32:
						CALL_KERNEL_IMPLEMENTATION(element_type::INT32)
					case element_type::INT64:
						CALL_KERNEL_IMPLEMENTATION(element_type::INT64)
					case element_type::UINT8:
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT8)
					case element_type::UINT16:
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT16)
					case element_type::UINT32:
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT32)
					case element_type::UINT64:
						CALL_KERNEL_IMPLEMENTATION(element_type::UINT64)
					case element_type::FLOAT32:
						CALL_KERNEL_IMPLEMENTATION(element_type::FLOAT32)
					case element_type::FLOAT64:
						CALL_KERNEL_IMPLEMENTATION(element_type::FLOAT64)
					}
#undef CALL_KERNEL_IMPLEMENTATION
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif