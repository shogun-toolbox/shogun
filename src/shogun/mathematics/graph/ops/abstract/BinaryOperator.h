/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNBINARYOPERATOR_H_
#define SHOGUNBINARYOPERATOR_H_

#include <shogun/mathematics/graph/ops/abstract/Operator.h>
#include <shogun/mathematics/graph/runtime/shogun/OutputNode.h>
#include <shogun/mathematics/graph/nodes/BinaryNode.h>

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
				ShogunBinaryOperator(const std::shared_ptr<node::Node>& node)
				    : Operator(node)
				{
				}

				virtual ~ShogunBinaryOperator()
				{
				}

				void call(const std::vector<std::shared_ptr<
				              detail::shogun::OutputNode>>& input_nodes) final
				{
					auto binary_node = std::static_pointer_cast<node::BaseBinaryNode>(m_node);
					const auto shape_compatibility = binary_node->get_binary_tensor_compatibility();

					if (input_nodes.size() != 2)
						error("Binary operation expected two inputs.");

					if (m_output_tensors.size() != 1)
						error("Binary operation expected one output.");

					const auto& input_tensor1 =
					    input_nodes[0]->get_output_tensors()[0];
					const auto& input_tensor2 =
					    input_nodes[1]->get_output_tensors()[0];
					const auto& output_tensor = m_output_tensors[0];

					runtime_checks_and_allocation(
					    input_tensor1, input_tensor2, shape_compatibility);
					if (shape_compatibility == node::BaseBinaryNode::BinaryShapeCompatibity::ArrayArray)
					{
						kernel(
						    input_tensor1->data(), input_tensor2->data(),
						    output_tensor->data(), output_tensor->size(),
						    input_tensor1->get_type());
					}
					else if (shape_compatibility == node::BaseBinaryNode::BinaryShapeCompatibity::ArrayScalar)
					{
						const bool scalar_first = input_tensor1->get_shape().is_scalar();
						kernel_scalar(
						    input_tensor1->data(), input_tensor2->data(),
						    output_tensor->data(), output_tensor->size(),
						    input_tensor1->get_type(), scalar_first);
					}
				}

			protected:
				void runtime_checks_and_allocation(
				    const std::shared_ptr<Tensor>& input_tensor1, 
				    const std::shared_ptr<Tensor>& input_tensor2,
				    const node::BaseBinaryNode::BinaryShapeCompatibity& shape_compatibility)
				{
					allocate_tensor(
					    runtime_shape_check(input_tensor1, input_tensor2, shape_compatibility));
				}

				void allocate_tensor(const Shape& shape)
				{
					m_output_tensors[0]->allocate_tensor(shape);
				}

				const Shape& runtime_shape_check(
				    const std::shared_ptr<Tensor>& tensor1,
				    const std::shared_ptr<Tensor>& tensor2,
				    const node::BaseBinaryNode::BinaryShapeCompatibity& shape_compatibility)
				{
					if (shape_compatibility == node::BaseBinaryNode::BinaryShapeCompatibity::ArrayArray)
					{
						for (auto [idx, shape1, shape2] :
						     enumerate(tensor1->get_shape(), tensor2->get_shape()))
						{
							if (shape1 != shape2)
							{
								error(
								    "Runtime shape mismatch in dimension {}. Got "
								    "{} and {}.",
								    idx, shape1, shape2);
							}
							if (shape1 == Shape::Dynamic)
							{
								error("Could not infer runtime shape.");
							}
						}
						// shapes have to match exactly so can return either one
						return tensor1->get_shape();
					}
					else if (shape_compatibility == node::BaseBinaryNode::BinaryShapeCompatibity::ArrayScalar)
					{
						if (tensor1->get_shape().is_scalar())
							return tensor2->get_shape();
						return tensor1->get_shape();
					}
					else
					{
						error("NotImplemented");
					}
				}

				void kernel(
				    void* input1, void* input2, void* output, const size_t size,
				    const element_type type)
				{

#define CALL_KERNEL_IMPLEMENTATION(SHOGUN_TYPE)                                \
	static_cast<DerivedOperator*>(this)                                        \
	    ->template kernel_implementation<                                      \
	        get_type_from_enum<SHOGUN_TYPE>::type>(                            \
	        input1, input2, output, size);                                     \
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

				void kernel_scalar(
				    void* input1, void* input2, void* output, const size_t size,
				    const element_type type, const bool scalar_first)
				{

#define CALL_KERNEL_IMPLEMENTATION(SHOGUN_TYPE)                                \
	static_cast<DerivedOperator*>(this)                                        \
	    ->template kernel_scalar_implementation<                               \
	        get_type_from_enum<SHOGUN_TYPE>::type>(                            \
	        input1, input2, output, size, scalar_first);                       \
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